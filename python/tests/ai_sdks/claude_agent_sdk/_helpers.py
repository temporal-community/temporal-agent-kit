"""Run an Agent Kit agent on the real Claude Code CLI, against a local mock of Anthropic's ``/v1/messages``.

Each test starts a Temporal dev server, a Worker with ``ClaudeAgentSDKPlugin(event_publisher=AgentKitEventPublisher)``
and ``AgentKitPlugin()``, and ``ClaudeTestAgent``, then talks to the agent through ``AgentClient`` as a user does.
The mock model's responder decides each model call from the request Claude Code sends it.
"""

from __future__ import annotations

import asyncio
import json
import signal
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Self, TypeVar

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)
from temporalio import workflow
from temporalio.client import WorkflowHandle
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from temporal_agent_kit import agent
from temporal_agent_kit.agent_client import AgentClient
from temporal_agent_kit.agent_protocol import (
    AgentConfig,
    AgentEvent,
    MessageHandlerEnd,
    MessageHandlerStart,
    MidTurn,
    TextMessage,
    TextReply,
    ToolApprovalPolicy,
    ToolApprovalRequested,
    ToolApprovalResolved,
    ToolEndEvent,
    ToolErrorEvent,
    ToolStartEvent,
)
from temporal_agent_kit.agent_workflow import (
    AgentWorkflowRunner,
    Injected,
    MessageContext,
)
from temporal_agent_kit.ai_sdks.claude_agent_sdk import (
    AgentKitClaudeSDKClient,
    AgentKitEventPublisher,
    ClaudeAgentSDKPlugin,
)
from temporal_agent_kit.ai_sdks.claude_agent_sdk._constants import (
    CLOSE_OPTIONS,
    CONNECT_OPTIONS,
    READ_OPTIONS,
    WRITE_OPTIONS,
)
from temporal_agent_kit.plugin import AgentKitPlugin

T = TypeVar("T")

TOOL_CANCELLED = "Tool call cancelled."
MODEL = "claude-test-model"

# Tools


@agent.tool_defn()
async def greet(name: str, prefix: Injected[str]) -> str:
    """Greet someone."""
    return f"{prefix} {name}"


@agent.tool_defn()
async def list_clothes(temperature: int) -> list[str]:
    """List the clothes to wear at a temperature."""
    return ["wool coat", "scarf"] if temperature < 10 else ["t-shirt"]


@agent.tool_defn()
async def explode() -> str:
    """Always fail."""
    raise RuntimeError("tool crashed")


@agent.tool_defn()
async def wait_a_while() -> str:
    """Take ten minutes."""
    await workflow.sleep(timedelta(minutes=10))
    return "waited"


RUN_CODE = agent.code_mode_tool([greet], name="run_code", inherently_safe=True, injections={"prefix": "Hello"})
AGENT_KIT_TOOLS = [greet, list_clothes, explode, wait_a_while, RUN_CODE]


@tool("echo", "Echo the text.", {"text": str})
async def _echo(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"echo: {args['text']}"}]}


OTHER_MCP_SERVERS: dict[str, Any] = {"other": create_sdk_mcp_server("other", tools=[_echo])}

# The agent


@dataclass
class ClientSetup:
    """How ``ClaudeTestAgent`` builds its ``AgentKitClaudeSDKClient``."""

    options: dict[str, Any] = field(default_factory=dict)
    """``ClaudeAgentOptions`` fields, over the defaults."""
    client_kwargs: dict[str, Any] = field(default_factory=dict)
    """Other ``AgentKitClaudeSDKClient`` arguments, such as transport ``hooks``."""

    def build(self, runner: AgentWorkflowRunner) -> AgentKitClaudeSDKClient:
        options = ClaudeAgentOptions(
            **{
                "setting_sources": [],
                "tools": ["Bash", "Read", "Write"],
                "mcp_servers": OTHER_MCP_SERVERS,
                **self.options,
            }
        )
        kwargs = {"tools": AGENT_KIT_TOOLS, "injections": {"prefix": "Hello"}, **self.client_kwargs}
        return AgentKitClaudeSDKClient(options, runner=runner, **kwargs)


_SETUPS: dict[str, ClientSetup] = {}  # By Workflow ID. The Worker runs Workflows unsandboxed, in the test's process.


@agent.defn(name="ClaudeTestAgent")
class ClaudeTestAgent:
    @agent.init
    def __init__(self, config: AgentConfig) -> None:
        self._runner = AgentWorkflowRunner(
            config, stream=WorkflowStream(), approval_policy_default=ToolApprovalPolicy.dangerously_skip_all()
        )
        self._client = _SETUPS[workflow.info().workflow_id].build(self._runner)

    @agent.setup
    async def connect(self) -> None:
        await self._client.connect()

    async def _ask(self, text: str) -> TextReply:
        await self._client.query(text)
        async for message in self._client.receive_response():
            if isinstance(message, ResultMessage):
                if message.is_error:
                    raise ApplicationError(message.result or "")
                return TextReply(text=message.result or "")
        raise ApplicationError("Claude Code ended without a result.")

    @agent.accepts(mid_turn=MidTurn.ENQUEUE)
    async def ask(self, message: TextMessage) -> TextReply:
        """Ask Claude something."""
        return await self._ask(message.text)

    @agent.accepts(mid_turn=MidTurn.ACCEPT)
    async def btw(self, message: TextMessage, ctx: Injected[MessageContext]) -> TextReply:
        """Steer Claude while it works, or ask it something while it is idle."""
        if ctx.joined_turn:
            self._runner.steer(message.text)
            return TextReply(text="")
        return await self._ask(message.text)

    @agent.teardown
    async def disconnect(self) -> None:
        await self._client.disconnect()


# The mock model


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def thinking(value: str, signature: str = "sig") -> dict[str, Any]:
    return {"type": "thinking", "thinking": value, "signature": signature}


def tool_use(tool_id: str, name: str, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input or {}}


def model_reply(
    *blocks: dict[str, Any],
    model: str = MODEL,
    usage: dict[str, Any] | None = None,
    output_tokens: int = 5,
) -> bytes:
    """An Anthropic ``/v1/messages`` SSE stream of one assistant message with ``blocks``."""
    stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    message = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 10, "output_tokens": 1},
    }
    events: list[dict[str, Any]] = [{"type": "message_start", "message": message}]
    for index, block in enumerate(blocks):
        match block["type"]:
            case "text":
                start, deltas = {"type": "text", "text": ""}, [{"type": "text_delta", "text": block["text"]}]
            case "thinking":
                start = {"type": "thinking", "thinking": "", "signature": ""}
                deltas = [
                    {"type": "thinking_delta", "thinking": block["thinking"]},
                    {"type": "signature_delta", "signature": block["signature"]},
                ]
            case "tool_use":
                start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
                deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
            case other:
                raise ValueError(other)
        events.append({"type": "content_block_start", "index": index, "content_block": start})
        events.extend({"type": "content_block_delta", "index": index, "delta": d} for d in deltas)
        events.append({"type": "content_block_stop", "index": index})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
    )
    events.append({"type": "message_stop"})
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


@dataclass
class ToolResult:
    content: str
    is_error: bool


def tool_results(body: dict[str, Any]) -> dict[str, ToolResult]:
    """The tool results in the last user message of a model request, by tool use ID."""
    content = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
    results: dict[str, ToolResult] = {}
    for block in content if isinstance(content, list) else []:
        if block.get("type") == "tool_result":
            value = block.get("content", "")
            if isinstance(value, list):
                value = "".join(part.get("text", "") for part in value)
            results[block["tool_use_id"]] = ToolResult(value, bool(block.get("is_error")))
    return results


def prompt_text(body: dict[str, Any]) -> str:
    """The text of the user's prompts in a model request, without Claude Code's context blocks."""
    parts: list[str] = []
    for message in body["messages"]:
        if message["role"] != "user":
            continue
        content = message["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        parts.extend(
            b["text"] for b in blocks if b.get("type") == "text" and not b["text"].lstrip().startswith(("<", "#"))
        )
    return "\n".join(parts)


Responder = Callable[[dict[str, Any]], bytes | Awaitable[bytes]]


def call_tools_then_reply(*calls: dict[str, Any]) -> Responder:
    """Call ``calls`` for each prompt, then reply with the results Claude Code sent back.

    Tool use IDs are unique, so each later call, such as a retried turn's, gets a ``_2``, ``_3``, ... suffix.
    """
    asked = 0

    def respond(body: dict[str, Any]) -> bytes:
        nonlocal asked
        results = tool_results(body)
        if results:
            return model_reply(text(json.dumps({k: [v.content, v.is_error] for k, v in sorted(results.items())})))
        asked += 1
        suffix = "" if asked == 1 else f"_{asked}"
        return model_reply(*({**c, "id": c["id"] + suffix} if c["type"] == "tool_use" else c for c in calls))

    return respond


def echo_prompt(body: dict[str, Any]) -> bytes:
    return model_reply(text(f"ECHO:{prompt_text(body).splitlines()[-1]}"))


class MockAnthropic:
    """A minimal HTTP/1.1 server speaking Anthropic's ``/v1/messages`` SSE protocol, recording each model request."""

    def __init__(self, responder: Responder) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        self._server: asyncio.Server | None = None
        self.url = ""

    async def __aenter__(self) -> Self:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *_args: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _read_request(self, reader: asyncio.StreamReader) -> tuple[str, bytes]:
        head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1").split("\r\n")
        headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in head[1:] if ":" in line)}
        if headers.get("transfer-encoding", "").lower() == "chunked":
            chunks: list[bytes] = []
            while size := int((await reader.readline()).strip().split(b";", 1)[0], 16):
                chunks.append(await reader.readexactly(size))
                await reader.readexactly(2)
            await reader.readline()
            return head[0], b"".join(chunks)
        length = int(headers.get("content-length", "0"))
        return head[0], await reader.readexactly(length) if length else b""

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line, raw = await self._read_request(reader)
            body = json.loads(raw) if raw else {}
            if "/v1/messages" in request_line and "count_tokens" not in request_line and body.get("stream", True):
                self.requests.append(body)
                result = self.responder(body)
                payload = await result if isinstance(result, Awaitable) else result
                content_type = "text/event-stream"
            elif "/v1/messages" in request_line and "count_tokens" not in request_line:
                # Claude Code's own side calls, such as naming the session.
                payload = json.dumps(
                    {
                        "id": "msg_side",
                        "type": "message",
                        "role": "assistant",
                        "model": MODEL,
                        "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode()
                content_type = "application/json"
            else:
                payload, content_type = b'{"input_tokens": 10}', "application/json"
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {len(payload)}\r\n"
                f"Connection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
        except Exception:  # noqa: BLE001, S110 - Claude Code retries a dropped request
            pass
        finally:
            writer.close()


# The session


def signal_claude(sig: signal.Signals) -> int:
    """Send ``sig`` to each running ``claude`` process, as a crash would. Returns how many it signaled."""
    from claude_agent_sdk._internal.transport.subprocess_cli import _ACTIVE_CHILDREN

    running = [p for p in list(_ACTIVE_CHILDREN) if p.returncode is None]
    for process in running:
        process.send_signal(sig)
    return len(running)


@dataclass
class Session:
    """An agent running on the real Claude Code CLI, and a user's client to it."""

    env: WorkflowEnvironment
    api: MockAnthropic
    handle: WorkflowHandle[Any, Any]
    client: AgentClient
    task_queue: str
    _worker: Worker
    _worker_task: asyncio.Task[None]

    async def turn(
        self, prompt: str, *, approve: bool | None = None, deny_reason: str = "No.", handler: str = "ask"
    ) -> list[AgentEvent]:
        """Send ``prompt`` and return the turn's events, answering each approval request with ``approve``."""
        return await self.collect(self.start_turn(prompt, handler=handler), approve=approve, deny_reason=deny_reason)

    def start_turn(self, prompt: str, *, handler: str = "ask") -> Any:
        return self.client.send_message(handler, {"text": prompt}, on_item=lambda item, _: item, timeout=60)

    async def collect(
        self,
        stream: Any,
        *,
        approve: bool | None = None,
        deny_reason: str = "No.",
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> list[AgentEvent]:
        """The events of a turn started with ``start_turn``, calling ``on_event`` on each."""
        events: list[AgentEvent] = []
        async for item in await stream:
            if not isinstance(item, AgentEvent):
                continue
            events.append(item)
            if on_event is not None:
                await on_event(item)
            if isinstance(item.event, ToolApprovalRequested) and approve is not None:
                await self.client.approve_tool(item.event.tool_id, approved=approve, reason=None if approve else deny_reason)
        return events

    async def steer(self, prompt: str) -> None:
        await self.client.submit_message("btw", {"text": prompt})

    async def restart_worker(self, *, before_start: Callable[[], None] | None = None) -> None:
        """Shut down the Worker, as when its process stops, and start a new one."""
        await self._worker.shutdown()
        await self._worker_task
        if before_start is not None:
            before_start()
        self._worker = _worker(self.env, self.task_queue)
        self._worker_task = asyncio.create_task(self._worker.run())


def _worker(env: WorkflowEnvironment, task_queue: str) -> Worker:
    return Worker(
        env.client,
        task_queue=task_queue,
        workflows=[ClaudeTestAgent],
        workflow_runner=UnsandboxedWorkflowRunner(),
        plugins=[ClaudeAgentSDKPlugin(event_publisher=AgentKitEventPublisher), AgentKitPlugin()],
    )


ASK_FIRST = AgentConfig(approval_policy=ToolApprovalPolicy.always_require_human_approval())
RUN_ALL = AgentConfig(approval_policy=ToolApprovalPolicy.dangerously_skip_all())


@asynccontextmanager
async def agent_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    responder: Responder = echo_prompt,
    *,
    config: AgentConfig = RUN_ALL,
    setup: ClientSetup | None = None,
) -> AsyncIterator[Session]:
    """Start ``ClaudeTestAgent`` on a Worker running the real Claude Code CLI against a mock model."""
    # Notice a lost Worker or Claude Code in seconds.
    for options in (CONNECT_OPTIONS, WRITE_OPTIONS, READ_OPTIONS, CLOSE_OPTIONS):
        monkeypatch.setitem(options, "schedule_to_start_timeout", timedelta(seconds=1))
        monkeypatch.setitem(options, "heartbeat_timeout", timedelta(seconds=2))

    async with MockAnthropic(responder) as api:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", api.url)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        monkeypatch.setenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MODE", "1")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
        monkeypatch.chdir(tmp_path)

        async with await WorkflowEnvironment.start_local() as env:
            task_queue = f"claude-test-{uuid.uuid4()}"
            _SETUPS[task_queue] = setup or ClientSetup()
            worker = _worker(env, task_queue)
            worker_task = asyncio.create_task(worker.run())
            handle = await env.client.start_workflow("ClaudeTestAgent", config, id=task_queue, task_queue=task_queue)
            session = Session(env, api, handle, AgentClient(env.client, handle.id), task_queue, worker, worker_task)
            try:
                yield session
            finally:
                with suppress(Exception):
                    await handle.terminate()
                await session._worker.shutdown()
                with suppress(Exception):
                    await session._worker_task
                del _SETUPS[task_queue]


# Reading a turn's events


def of_type(events: list[AgentEvent], event_type: type[T]) -> list[T]:
    return [e.event for e in events if isinstance(e.event, event_type)]


def reply_text(events: list[AgentEvent]) -> str:
    """The reply to the message that started the turn, not to steering that joined it."""
    first = next(e.message_id for e in events if isinstance(e.event, MessageHandlerStart))
    (end,) = [e.event for e in events if isinstance(e.event, MessageHandlerEnd) and e.message_id == first]
    return TextReply.model_validate(end.output).text


def tool_outcomes(events: list[AgentEvent]) -> list[ToolEndEvent | ToolErrorEvent]:
    return [e.event for e in events if isinstance(e.event, (ToolEndEvent, ToolErrorEvent))]


def assert_tools_settled(events: list[AgentEvent]) -> None:
    """Each tool that started ended or failed once, and each approval request was answered once."""
    started = [e.tool_id for e in of_type(events, ToolStartEvent)]
    settled = [e.tool_id for e in tool_outcomes(events)]
    assert set(started) <= set(settled), (started, settled)
    assert len(settled) == len(set(settled)), settled
    asked = [e.tool_id for e in of_type(events, ToolApprovalRequested)]
    answered = [e.tool_id for e in of_type(events, ToolApprovalResolved)]
    assert sorted(asked) == sorted(answered), (asked, answered)
