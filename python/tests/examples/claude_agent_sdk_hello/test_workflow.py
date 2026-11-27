"""End-to-end Temporal server + real Claude Code CLI binary tests for ``ClaudeAgentSDKHelloWorkflow``.

Runs the actual bundled ``claude`` CLI binary via ``ClaudeAgentSDKPlugin(event_publisher=AgentKitEventPublisher)``
(``SubprocessConnector`` — no fake connectors or fake transports) against a local mock
Anthropic ``/v1/messages`` SSE server, using ``AgentClient`` interfaces.

1. ``test_full_session_lifecycle`` — one long-running workflow exercising all healthy
   scenarios in sequence on a single real Claude Code process:
   - plain LLM turn
   - built-in tool (Write) approved, then denied, by human
   - Agent Kit MCP tool (run_code) denied by human
   - Code Mode (Agent Kit MCP run_code) chaining inner Agent Kit tools with human approval
   - steering claimed mid-tool via PostToolUse
   - steering queued during an LLM-only turn and dropped at turn end
   - clean close

2. Four isolated disaster recovery tests (the real failure modes in production):
   - worker_soft_crash (SIGTERM Claude + Worker 1 shutdown -> Worker 2 recovery)
   - worker_hard_crash (Worker 1 shutdown -> Worker 2 recovery)
   - claude_soft_crash (SIGTERM real claude OS process -> fresh process recovery)
   - claude_hard_crash (SIGKILL real claude OS process -> fresh process recovery)
"""

from __future__ import annotations

import asyncio
import json
import signal
import tempfile
from collections.abc import Callable
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from examples.claude_agent_sdk_hello.workflow import ClaudeAgentSDKHelloWorkflow
from temporal_agent_kit.agent_client import AgentClient
from temporal_agent_kit.agent_protocol import (
    AgentConfig,
    AgentEvent,
    AgentEventType,
    MessageHandlerEnd,
    ModelInteractionEnded,
    ModelInteractionStarted,
    ReplyDelta,
    TextReply,
    ToolApprovalPolicy,
    ToolApprovalResolved,
    ToolEndEvent,
    ToolErrorEvent,
    ToolRequested,
    ToolStartEvent,
)
from temporal_agent_kit.ai_sdks.claude_agent_sdk import AgentKitEventPublisher, ClaudeAgentSDKPlugin
from temporal_agent_kit.ai_sdks.claude_agent_sdk._constants import (
    CLOSE_OPTIONS,
    CONNECT_OPTIONS,
    READ_OPTIONS,
    WRITE_OPTIONS,
)
from temporal_agent_kit.plugin import AgentKitPlugin

_CODE_SCRIPT = """\
import asyncio
async def main():
    temperature = await get_temperature("Toronto")
    prices = {}
    for item in await recommend_clothing(temperature):
        prices[item] = await get_price(item)
    return prices
asyncio.run(main())
"""

_AUTO_APPROVE_ALL = AgentConfig(approval_policy=ToolApprovalPolicy.dangerously_skip_all())


# ---------------------------------------------------------------------------
# Local mock Anthropic /v1/messages SSE server
# ---------------------------------------------------------------------------


def _sse_text(text: str, *, msg_id: str = "msg_1") -> bytes:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events).encode()


def _sse_tool_use(tool_id: str, name: str, tool_input: dict[str, Any], *, msg_id: str = "msg_tool") -> bytes:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input)},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 10}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events).encode()


class _MockAnthropicServer:
    """Minimal HTTP/1.1 server speaking Anthropic's ``/v1/messages`` SSE protocol."""

    def __init__(self, responder: Callable[[dict[str, Any]], bytes | asyncio.Future[bytes]]) -> None:
        self._responder = responder
        self._server: asyncio.Server | None = None
        self.url: str = ""

    async def __aenter__(self) -> _MockAnthropicServer:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _read_http_request(self, reader: asyncio.StreamReader) -> tuple[str, bytes]:
        raw_head = await reader.readuntil(b"\r\n\r\n")
        head_lines = raw_head.decode("latin-1").split("\r\n")
        request_line = head_lines[0]
        headers: dict[str, str] = {}
        for line in head_lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        if headers.get("transfer-encoding", "").lower() == "chunked":
            chunks: list[bytes] = []
            while True:
                size_line = await reader.readline()
                size = int(size_line.strip().split(b";", 1)[0], 16)
                if size == 0:
                    await reader.readline()
                    break
                chunks.append(await reader.readexactly(size))
                await reader.readexactly(2)
            return request_line, b"".join(chunks)

        length = int(headers.get("content-length", "0"))
        body = await reader.readexactly(length) if length > 0 else b""
        return request_line, body

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line, body = await self._read_http_request(reader)
            if "/v1/messages" in request_line and "count_tokens" not in request_line:
                payload = json.loads(body.decode("utf-8")) if body else {}
                if payload.get("stream") is False:
                    resp = json.dumps({
                        "id": "msg_sync",
                        "type": "message",
                        "role": "assistant",
                        "model": payload.get("model", "claude-sonnet-4-5-20250929"),
                        "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }).encode()
                    header = (
                        f"HTTP/1.1 200 OK\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Length: {len(resp)}\r\n"
                        f"Connection: close\r\n\r\n"
                    ).encode()
                    writer.write(header + resp)
                else:
                    result = self._responder(payload)
                    sse = await result if asyncio.isfuture(result) or asyncio.iscoroutine(result) else result
                    header = (
                        f"HTTP/1.1 200 OK\r\n"
                        f"Content-Type: text/event-stream\r\n"
                        f"Content-Length: {len(sse)}\r\n"
                        f"Connection: close\r\n\r\n"
                    ).encode()
                    writer.write(header + sse)
            else:
                ok = b'{"input_tokens": 10}'
                header = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(ok)}\r\n"
                    f"Connection: close\r\n\r\n"
                ).encode()
                writer.write(header + ok)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()


def _configure_mock_api_env(monkeypatch: pytest.MonkeyPatch, base_url: str, config_dir: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MODE", "1")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))


def _patch_fast_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    for opts in (CONNECT_OPTIONS, WRITE_OPTIONS, READ_OPTIONS, CLOSE_OPTIONS):
        monkeypatch.setitem(opts, "schedule_to_start_timeout", timedelta(seconds=1))
        monkeypatch.setitem(opts, "heartbeat_timeout", timedelta(seconds=2))


def _signal_claude_children(sig: signal.Signals) -> int:
    """Send ``sig`` to any live ``claude`` child process spawned by ``SubprocessCLITransport``."""
    from claude_agent_sdk._internal.transport.subprocess_cli import _ACTIVE_CHILDREN

    signalled = 0
    for proc in list(_ACTIVE_CHILDREN):
        if proc.returncode is None:
            proc.send_signal(sig)
            signalled += 1
    return signalled


def _events_of(events: list[AgentEvent], event_type: AgentEventType) -> list[AgentEvent]:
    return [e for e in events if e.event.type == event_type]


def _assert_each_model_tool_call_requested_first(events: list[AgentEvent]) -> None:
    """Each tool call the model made has one ToolRequested, published before its other events."""
    by_tool: dict[str, list[Any]] = {}
    for ev in events:
        tool_id = getattr(ev.event, "tool_id", None)
        if tool_id is not None and tool_id.startswith("toolu_"):
            by_tool.setdefault(tool_id, []).append(ev.event)
    assert by_tool, [e.event for e in events]
    for tool_id, tool_events in by_tool.items():
        assert isinstance(tool_events[0], ToolRequested), (tool_id, tool_events)
        assert sum(isinstance(e, ToolRequested) for e in tool_events) == 1, (tool_id, tool_events)


def _reply_text(events: list[AgentEvent]) -> str:
    for ev in reversed(events):
        if isinstance(ev.event, MessageHandlerEnd):
            text = TextReply.model_validate(ev.event.output).text
            if text:
                return text
    return ""


def _last_user_text(body: dict[str, Any]) -> str:
    """Return all user/tool_result text after the last assistant message in ``/v1/messages``."""
    parts: list[str] = []
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") != "user":
            if parts:
                break
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if not content.startswith("# Environment"):
                parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    parts.append(f"TOOL_RESULT:{block.get('content', '')}")
                elif block.get("type") == "text":
                    text = block.get("text", "")
                    if not text.startswith("# Environment"):
                        parts.append(text)
    return "\n".join(reversed(parts))


async def _send_turn(
    client: AgentClient,
    text: str,
    *,
    approve: bool | None = None,
    deny_reason: str = "Too risky",
    timeout: float = 30,
) -> list[AgentEvent]:
    """Send one ``ask`` turn via ``AgentClient.send_message`` and optionally answer approval gates."""
    collected: list[AgentEvent] = []
    stream = await client.send_message(
        ClaudeAgentSDKHelloWorkflow.ask.__name__,
        {"text": text},
        on_item=lambda item, _: item,
        timeout=timeout,
    )
    async for item in stream:
        if not isinstance(item, AgentEvent):
            continue
        collected.append(item)
        if item.event.type == AgentEventType.TOOL_APPROVAL_REQUESTED and approve is not None:
            await client.approve_tool(
                item.event.tool_id,
                approved=approve,
                reason=None if approve else deny_reason,
            )
    return collected


# ---------------------------------------------------------------------------
# 1. Single long-running workflow exercising every healthy scenario on real CLI
# ---------------------------------------------------------------------------


async def test_full_session_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    approved_file = tmp_path / "approved.txt"
    denied_file = tmp_path / "denied.txt"
    hold_turn = asyncio.Event()
    turn_reached_llm = asyncio.Event()
    hold_on_prompts: set[str] = {"Run steered tool.", "Write a poem."}
    call_count = 0

    async def respond(body: dict[str, Any]) -> bytes:
        nonlocal call_count
        call_count += 1
        text = _last_user_text(body)

        if "TOOL_RESULT:" in text:
            steered = "|STEERED" if "Make it short." in json.dumps(body) else ""
            return _sse_text(f"AFTER_TOOL:{text}{steered}", msg_id=f"msg_{call_count}")

        if any(p in text for p in hold_on_prompts):
            turn_reached_llm.set()
            await hold_turn.wait()
            hold_turn.clear()

        if "Approve write." in text:
            return _sse_tool_use(
                "toolu_write_approve",
                "Write",
                {"file_path": str(approved_file), "content": "approved"},
                msg_id=f"msg_{call_count}",
            )
        if "Deny write." in text:
            return _sse_tool_use(
                "toolu_write_deny",
                "Write",
                {"file_path": str(denied_file), "content": "denied"},
                msg_id=f"msg_{call_count}",
            )
        if "Deny code mode." in text:
            return _sse_tool_use(
                "toolu_code_deny",
                "mcp__agent_kit__run_code",
                {"script": "await get_temperature('Toronto')"},
                msg_id=f"msg_{call_count}",
            )
        if "Run code mode." in text:
            return _sse_tool_use(
                "toolu_code_1",
                "mcp__agent_kit__run_code",
                {"script": _CODE_SCRIPT},
                msg_id=f"msg_{call_count}",
            )
        if "Run steered tool." in text:
            return _sse_tool_use(
                "toolu_code_steer",
                "mcp__agent_kit__run_code",
                {"script": "await get_temperature('Toronto')"},
                msg_id=f"msg_{call_count}",
            )
        return _sse_text(f"ECHO:{text.strip()}", msg_id=f"msg_{call_count}")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with _MockAnthropicServer(respond) as api:
            _configure_mock_api_env(monkeypatch, api.url, Path(tmpdir))
            async with await WorkflowEnvironment.start_local() as env:
                task_queue = str(uuid4())
                async with Worker(
                    env.client,
                    task_queue=task_queue,
                    workflows=[ClaudeAgentSDKHelloWorkflow],
                    plugins=[ClaudeAgentSDKPlugin(event_publisher=AgentKitEventPublisher), AgentKitPlugin()],
                    # Evict after every task, so the workflow replays while approvals are pending.
                    max_cached_workflows=0,
                ):
                    handle = await env.client.start_workflow(
                        "ClaudeAgentSDKHello",
                        AgentConfig(approval_policy=ToolApprovalPolicy.always_require_human_approval()),
                        id=task_queue,
                        task_queue=task_queue,
                    )
                    client = AgentClient(env.client, handle.id)

                    try:
                        # Turn 1: Plain LLM call
                        ev1 = await _send_turn(client, "Hello real Claude")
                        assert "Hello real Claude" in _reply_text(ev1)
                        assert any(isinstance(e.event, ModelInteractionStarted) for e in ev1)
                        assert any(isinstance(e.event, ModelInteractionEnded) for e in ev1)
                        assert any(isinstance(e.event, ReplyDelta) for e in ev1)

                        # Turn 2: Built-in tool (Write) approved by human, then denied
                        ev2 = await _send_turn(client, "Approve write.", approve=True)
                        assert "AFTER_TOOL:" in _reply_text(ev2)
                        assert [e.event.approved for e in ev2 if isinstance(e.event, ToolApprovalResolved)] == [True]
                        assert any(
                            isinstance(e.event, ToolEndEvent) and e.event.tool_name == "Write" for e in ev2
                        ), [e.event for e in ev2]
                        assert approved_file.read_text() == "approved"
                        _assert_each_model_tool_call_requested_first(ev2)

                        ev2_deny = await _send_turn(client, "Deny write.", approve=False, deny_reason="No writes")
                        assert "No writes" in _reply_text(ev2_deny)
                        assert [
                            e.event.approved for e in ev2_deny if isinstance(e.event, ToolApprovalResolved)
                        ] == [False]
                        assert not denied_file.exists()

                        # Turn 3: Agent Kit MCP tool (run_code) denied by human
                        ev3 = await _send_turn(client, "Deny code mode.", approve=False, deny_reason="Too risky")
                        assert "Too risky" in _reply_text(ev3)
                        assert [e.event.approved for e in ev3 if isinstance(e.event, ToolApprovalResolved)] == [False]
                        assert [e for e in ev3 if isinstance(e.event, ToolStartEvent)] == [], [e.event for e in ev3]
                        assert [e for e in ev3 if isinstance(e.event, ToolErrorEvent)] == []
                        _assert_each_model_tool_call_requested_first(ev3)

                        # Turn 4: Code Mode (Agent Kit MCP run_code) + inner Agent Kit tools approved by human
                        ev4 = await _send_turn(client, "Run code mode.", approve=True)
                        expected_code_result = "result: {'wool coat': 49.99, 'scarf': 49.99, 'boots': 49.99}"
                        assert expected_code_result in _reply_text(ev4)
                        assert [
                            (e.event.tool_name, e.event.tool_output)
                            for e in ev4
                            if isinstance(e.event, ToolEndEvent)
                        ] == [
                            ("get_temperature", 8),
                            ("recommend_clothing", ["wool coat", "scarf", "boots"]),
                            ("get_price", 49.99),
                            ("get_price", 49.99),
                            ("get_price", 49.99),
                            ("run_code", expected_code_result),
                        ]
                        assert any(
                            isinstance(e.event, ToolRequested) and e.event.tool_name == "run_code" for e in ev4
                        )
                        _assert_each_model_tool_call_requested_first(ev4)

                        # Turn 5: Steering claimed mid-tool (PostToolUse)
                        turn_reached_llm.clear()
                        turn5_task = asyncio.create_task(_send_turn(client, "Run steered tool.", approve=True))
                        await turn_reached_llm.wait()
                        await client.submit_message(
                            ClaudeAgentSDKHelloWorkflow.btw.__name__,
                            {"text": "Make it short."},
                        )
                        hold_turn.set()
                        ev5 = await turn5_task
                        (enqueued5,) = _events_of(ev5, AgentEventType.STEERING_ENQUEUED)
                        (claimed5,) = _events_of(ev5, AgentEventType.STEERING_CLAIMED)
                        assert claimed5.event.steering_ids == [enqueued5.event.steering_id]
                        assert not _events_of(ev5, AgentEventType.STEERING_DROPPED)
                        assert "|STEERED" in _reply_text(ev5)

                        # Turn 6: Steering queued during LLM-only turn -> dropped at turn end
                        turn_reached_llm.clear()
                        turn6_task = asyncio.create_task(_send_turn(client, "Write a poem."))
                        await turn_reached_llm.wait()
                        await client.submit_message(
                            ClaudeAgentSDKHelloWorkflow.btw.__name__,
                            {"text": "Too late."},
                        )
                        hold_turn.set()
                        ev6 = await turn6_task
                        (enqueued6,) = _events_of(ev6, AgentEventType.STEERING_ENQUEUED)
                        (dropped6,) = _events_of(ev6, AgentEventType.STEERING_DROPPED)
                        assert dropped6.event.steering_ids == [enqueued6.event.steering_id]
                        assert not _events_of(ev6, AgentEventType.STEERING_CLAIMED)
                        assert "Write a poem." in _reply_text(ev6)

                        # Clean close
                        await handle.signal("close")
                        await handle.result()

                        # With no cache, this query replays the whole history; it fails on non-determinism.
                        await client.get_status()
                    finally:
                        with suppress(Exception):
                            await handle.terminate()


# ---------------------------------------------------------------------------
# 2. Isolated end-to-end disaster recovery tests (real Temporal + real CLI)
# ---------------------------------------------------------------------------


async def _run_real_claude_signal_disaster(
    monkeypatch: pytest.MonkeyPatch,
    sig: signal.Signals,
) -> list[AgentEvent]:
    """Kill the real ``claude`` OS process mid-turn with ``SIGTERM`` or ``SIGKILL`` and verify recovery."""
    _patch_fast_timeouts(monkeypatch)
    first_request = asyncio.Event()
    release_first = asyncio.Event()
    request_num = 0

    async def respond(body: dict[str, Any]) -> bytes:
        nonlocal request_num
        request_num += 1
        if request_num == 1:
            first_request.set()
            await release_first.wait()
        return _sse_text("Recovered after signal.", msg_id=f"msg_{request_num}")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with _MockAnthropicServer(respond) as api:
            _configure_mock_api_env(monkeypatch, api.url, Path(tmpdir))
            async with await WorkflowEnvironment.start_local() as env:
                task_queue = str(uuid4())
                async with Worker(
                    env.client,
                    task_queue=task_queue,
                    workflows=[ClaudeAgentSDKHelloWorkflow],
                    plugins=[ClaudeAgentSDKPlugin(event_publisher=AgentKitEventPublisher), AgentKitPlugin()],
                ):
                    handle = await env.client.start_workflow(
                        "ClaudeAgentSDKHello", _AUTO_APPROVE_ALL, id=task_queue, task_queue=task_queue
                    )
                    client = AgentClient(env.client, handle.id)
                    try:
                        turn_task = asyncio.create_task(_send_turn(client, "Crash claude.", timeout=20))
                        await first_request.wait()
                        assert _signal_claude_children(sig) >= 1
                        release_first.set()

                        collected = await turn_task
                        assert _reply_text(collected) == "Recovered after signal."
                        return collected
                    finally:
                        release_first.set()
                        with suppress(Exception):
                            await handle.terminate()


async def _run_real_worker_crash_disaster(
    monkeypatch: pytest.MonkeyPatch,
    *,
    soft_signal_claude: bool,
) -> list[AgentEvent]:
    """Shut down Worker 1 mid-turn while running the real ``claude`` binary, then recover on Worker 2."""
    _patch_fast_timeouts(monkeypatch)
    first_request = asyncio.Event()
    release_first = asyncio.Event()
    request_num = 0

    async def respond(body: dict[str, Any]) -> bytes:
        nonlocal request_num
        request_num += 1
        if request_num == 1:
            first_request.set()
            await release_first.wait()
        return _sse_text("Recovered on worker 2.", msg_id=f"msg_{request_num}")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with _MockAnthropicServer(respond) as api:
            _configure_mock_api_env(monkeypatch, api.url, Path(tmpdir))
            async with await WorkflowEnvironment.start_local() as env:
                task_queue = str(uuid4())
                worker_1 = Worker(
                    env.client,
                    task_queue=task_queue,
                    workflows=[ClaudeAgentSDKHelloWorkflow],
                    plugins=[ClaudeAgentSDKPlugin(event_publisher=AgentKitEventPublisher), AgentKitPlugin()],
                )
                w1_task = asyncio.create_task(worker_1.run())
                handle = await env.client.start_workflow(
                    "ClaudeAgentSDKHello", _AUTO_APPROVE_ALL, id=task_queue, task_queue=task_queue
                )
                client = AgentClient(env.client, handle.id)
                try:
                    turn_task = asyncio.create_task(_send_turn(client, "Crash worker.", timeout=30))
                    await first_request.wait()
                    if soft_signal_claude:
                        _signal_claude_children(signal.SIGTERM)
                    release_first.set()

                    await worker_1.shutdown()
                    await w1_task

                    worker_2 = Worker(
                        env.client,
                        task_queue=task_queue,
                        workflows=[ClaudeAgentSDKHelloWorkflow],
                        plugins=[ClaudeAgentSDKPlugin(event_publisher=AgentKitEventPublisher), AgentKitPlugin()],
                    )
                    w2_task = asyncio.create_task(worker_2.run())
                    try:
                        collected = await turn_task
                    finally:
                        await worker_2.shutdown()
                        await w2_task

                    assert _reply_text(collected) == "Recovered on worker 2."
                    return collected
                finally:
                    release_first.set()
                    if not w1_task.done():
                        await worker_1.shutdown()
                        await w1_task
                    with suppress(Exception):
                        await handle.terminate()


async def test_disaster_worker_soft_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    collected = await _run_real_worker_crash_disaster(monkeypatch, soft_signal_claude=True)
    assert any(isinstance(e.event, ModelInteractionEnded) for e in collected)


async def test_disaster_worker_hard_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    collected = await _run_real_worker_crash_disaster(monkeypatch, soft_signal_claude=False)
    assert any(isinstance(e.event, ModelInteractionEnded) for e in collected)


async def test_disaster_claude_soft_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    collected = await _run_real_claude_signal_disaster(monkeypatch, signal.SIGTERM)
    assert any(isinstance(e.event, ModelInteractionEnded) for e in collected)


async def test_disaster_claude_hard_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    collected = await _run_real_claude_signal_disaster(monkeypatch, signal.SIGKILL)
    assert any(isinstance(e.event, ModelInteractionEnded) for e in collected)
