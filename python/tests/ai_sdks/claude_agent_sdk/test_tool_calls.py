"""Tool calls: built-in tools, Agent Kit tools, Code Mode, and other MCP servers' tools, as a user sees them run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from temporalio.client import WorkflowFailureError

from temporal_agent_kit.agent_protocol import (
    ToolApprovalRequested,
    ToolEndEvent,
    ToolErrorEvent,
    ToolRequested,
    ToolStartEvent,
)
from tests.ai_sdks.claude_agent_sdk._helpers import (
    ASK_FIRST,
    OTHER_MCP_SERVERS,
    ClientSetup,
    agent_session,
    assert_tools_settled,
    call_tools_then_reply,
    of_type,
    tool_outcomes,
    tool_results,
    tool_use,
)


@dataclass(frozen=True)
class Tool:
    """A tool call the model makes, and what the user and the model see of it."""

    tool_id: str
    claude_name: str
    agent_kit_name: str
    tool_input: dict[str, Any]
    result: str  # What the model gets back


# Writes a file, so Claude Code asks permission to run it.
BUILT_IN = Tool("toolu_bash", "Bash", "Bash", {"command": "printf built-in | tee out.txt"}, "built-in")
AGENT_KIT = Tool("toolu_greet", "mcp__agent_kit__greet", "greet", {"name": "Claude"}, "Hello Claude")
CODE_MODE = Tool(
    "toolu_code", "mcp__agent_kit__run_code", "run_code", {"script": "await greet(name='Code')"}, "Hello Code"
)

TOOLS = [
    pytest.param((BUILT_IN,), id="built_in"),
    pytest.param((AGENT_KIT,), id="agent_kit"),
    pytest.param((CODE_MODE,), id="code_mode"),
    pytest.param((BUILT_IN, AGENT_KIT, CODE_MODE), id="all_parallel"),
]


def _calls(*tools: Tool) -> list[dict[str, Any]]:
    return [tool_use(t.tool_id, t.claude_name, t.tool_input) for t in tools]


@pytest.mark.parametrize("tools", TOOLS)
async def test_each_tool_runs_and_its_result_reaches_the_model(
    tools: tuple[Tool, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with agent_session(monkeypatch, tmp_path, call_tools_then_reply(*_calls(*tools))) as session:
        events = await session.turn("Use the tools")
        results = tool_results(session.api.requests[-1])

    # Claude Code sends the model's tool calls before it runs them, named as Agent Kit knows them.
    assert of_type(events, ToolRequested) == [
        ToolRequested(tool_id=t.tool_id, tool_name=t.agent_kit_name, tool_input=t.tool_input) for t in tools
    ]
    called = {t.tool_id for t in tools}  # Code Mode also reports the tools its script calls
    assert {e.tool_id: e.tool_name for e in of_type(events, ToolStartEvent) if e.tool_id in called} == {
        t.tool_id: t.agent_kit_name for t in tools
    }
    assert {e.tool_id for e in of_type(events, ToolEndEvent)} >= called
    assert_tools_settled(events)
    assert {tool_id: (r.is_error, r.content.strip()) for tool_id, r in results.items()} == {
        t.tool_id: (False, t.result) if t is not CODE_MODE else (False, f"result: {t.result!r}") for t in tools
    }


async def test_a_built_in_tools_end_carries_its_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("remember the milk")
    read = tool_use("toolu_read", "Read", {"file_path": str(tmp_path / "notes.txt")})
    async with agent_session(monkeypatch, tmp_path, call_tools_then_reply(read)) as session:
        events = await session.turn("Read my notes")

    (end,) = of_type(events, ToolEndEvent)
    assert (end.tool_id, end.tool_name) == ("toolu_read", "Read")
    assert "remember the milk" in str(end.tool_output)


@pytest.mark.parametrize(
    ("call", "tool_name"),
    [
        pytest.param(tool_use("toolu_read", "Read", {"file_path": "/no/such/file.txt"}), "Read", id="built_in"),
        pytest.param(tool_use("toolu_explode", "mcp__agent_kit__explode"), "explode", id="agent_kit"),
    ],
)
async def test_a_failed_tool_reports_its_error_to_the_user_and_the_model(
    call: dict[str, Any], tool_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with agent_session(monkeypatch, tmp_path, call_tools_then_reply(call)) as session:
        events = await session.turn("Try it")
        result = tool_results(session.api.requests[-1])[call["id"]]

    (error,) = tool_outcomes(events)
    assert isinstance(error, ToolErrorEvent)
    assert (error.tool_id, error.tool_name) == (call["id"], tool_name)
    assert result.is_error


@pytest.mark.parametrize(
    ("temperature", "text"),
    [pytest.param(5, '["wool coat","scarf"]', id="list"), pytest.param(20, '["t-shirt"]', id="one_item")],
)
async def test_a_structured_tool_result_reaches_the_model_as_json(
    temperature: int, text: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call = tool_use("toolu_clothes", "mcp__agent_kit__list_clothes", {"temperature": temperature})
    async with agent_session(monkeypatch, tmp_path, call_tools_then_reply(call)) as session:
        await session.turn("What should I wear?")
        result = tool_results(session.api.requests[-1])["toolu_clothes"]

    assert (result.is_error, result.content) == (False, text)


async def test_the_model_sees_agent_kit_tools_without_their_injected_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with agent_session(monkeypatch, tmp_path) as session:
        await session.turn("Hi")
        tools = {t["name"]: t for t in session.api.requests[-1]["tools"]}

    assert tools["mcp__agent_kit__greet"]["input_schema"]["properties"].keys() == {"name"}
    assert tools["mcp__agent_kit__run_code"]["input_schema"]["properties"].keys() == {"script"}
    assert "mcp__other__echo" in tools


async def test_another_mcp_servers_tool_keeps_its_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    echo = tool_use("toolu_echo", "mcp__other__echo", {"text": "hi"})
    async with agent_session(monkeypatch, tmp_path, call_tools_then_reply(echo)) as session:
        events = await session.turn("Echo hi")
        result = tool_results(session.api.requests[-1])["toolu_echo"]

    assert of_type(events, ToolRequested) == [
        ToolRequested(tool_id="toolu_echo", tool_name="mcp__other__echo", tool_input={"text": "hi"})
    ]
    assert [(type(e), e.tool_name) for e in [*of_type(events, ToolStartEvent), *tool_outcomes(events)]] == [
        (ToolStartEvent, "mcp__other__echo"),
        (ToolEndEvent, "mcp__other__echo"),
    ]
    assert result.content == "echo: hi"


async def test_a_tool_claude_code_runs_without_asking_still_reports_its_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Claude Code runs the tools in allowed_tools without asking for permission.
    (tmp_path / "notes.txt").write_text("notes")
    read = tool_use("toolu_read", "Read", {"file_path": str(tmp_path / "notes.txt")})
    setup = ClientSetup(options={"allowed_tools": ["Read"]})
    async with agent_session(monkeypatch, tmp_path, call_tools_then_reply(read), config=ASK_FIRST, setup=setup) as session:
        events = await session.turn("Read my notes")

    assert of_type(events, ToolApprovalRequested) == []
    assert [(type(e), e.tool_id) for e in tool_outcomes(events)] == [(ToolEndEvent, "toolu_read")]
    assert_tools_settled(events)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        pytest.param({"mcp_servers": {"agent_kit": OTHER_MCP_SERVERS["other"]}}, "reserved", id="reserved_name"),
        pytest.param({"mcp_servers": "mcp.json"}, "only supported as a dict", id="config_file"),
    ],
)
async def test_an_agent_with_tools_needs_its_own_mcp_server_name(
    options: dict[str, Any], message: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with agent_session(monkeypatch, tmp_path, setup=ClientSetup(options=options)) as session:
        with pytest.raises(WorkflowFailureError) as failure:
            await session.handle.result()

    assert message in str(failure.value.cause)
