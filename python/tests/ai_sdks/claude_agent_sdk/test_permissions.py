"""Permissions: which tool calls ask the user for approval, and what happens when the user answers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import PermissionResultDeny, ToolPermissionContext

from temporal_agent_kit.agent_protocol import (
    ToolApprovalRequested,
    ToolApprovalResolved,
    ToolEndEvent,
    ToolRequested,
    ToolStartEvent,
)
from tests.ai_sdks.claude_agent_sdk._helpers import (
    ASK_FIRST,
    ClientSetup,
    agent_session,
    assert_tools_settled,
    call_tools_then_reply,
    of_type,
    tool_results,
    tool_use,
)


def _write(tmp_path: Path) -> dict[str, Any]:
    return tool_use("toolu_write", "Write", {"file_path": str(tmp_path / "out.txt"), "content": "written"})


async def test_an_approved_built_in_tool_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with agent_session(monkeypatch, tmp_path, call_tools_then_reply(_write(tmp_path)), config=ASK_FIRST) as session:
        events = await session.turn("Write a file", approve=True)

    assert [(e.tool_id, e.tool_name) for e in of_type(events, ToolApprovalRequested)] == [("toolu_write", "Write")]
    assert [(e.tool_id, e.approved) for e in of_type(events, ToolApprovalResolved)] == [("toolu_write", True)]
    assert [e.tool_id for e in of_type(events, ToolStartEvent)] == ["toolu_write"]
    assert [e.tool_id for e in of_type(events, ToolEndEvent)] == ["toolu_write"]
    assert (tmp_path / "out.txt").read_text() == "written"


async def test_a_denied_built_in_tool_never_runs_and_the_model_hears_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with agent_session(monkeypatch, tmp_path, call_tools_then_reply(_write(tmp_path)), config=ASK_FIRST) as session:
        events = await session.turn("Write a file", approve=False, deny_reason="Not today.")
        result = tool_results(session.api.requests[-1])["toolu_write"]

    assert [(e.tool_id, e.approved) for e in of_type(events, ToolApprovalResolved)] == [("toolu_write", False)]
    assert of_type(events, ToolStartEvent) == []
    assert not (tmp_path / "out.txt").exists()
    assert result.is_error and "Not today." in result.content
    assert_tools_settled(events)


@pytest.mark.parametrize(
    ("call", "tool_name"),
    [
        pytest.param(tool_use("toolu_greet", "mcp__agent_kit__greet", {"name": "Claude"}), "greet", id="agent_kit"),
        pytest.param(
            tool_use("toolu_code", "mcp__agent_kit__run_code", {"script": "await greet(name='Claude')"}),
            "run_code",
            id="code_mode",
        ),
        pytest.param(tool_use("toolu_echo", "mcp__other__echo", {"text": "hi"}), "mcp__other__echo", id="other_mcp"),
    ],
)
async def test_an_mcp_tool_asks_for_approval_once(
    call: dict[str, Any], tool_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with agent_session(monkeypatch, tmp_path, call_tools_then_reply(call), config=ASK_FIRST) as session:
        events = await session.turn("Use the tool", approve=True)

    assert [e.tool_name for e in of_type(events, ToolApprovalRequested) if e.tool_id == call["id"]] == [tool_name]
    assert call["id"] in {e.tool_id for e in of_type(events, ToolEndEvent)}
    assert_tools_settled(events)


async def test_the_callers_permission_mode_is_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # acceptEdits lets Claude Code write files without asking.
    setup = ClientSetup(options={"permission_mode": "acceptEdits"})
    async with agent_session(
        monkeypatch, tmp_path, call_tools_then_reply(_write(tmp_path)), config=ASK_FIRST, setup=setup
    ) as session:
        events = await session.turn("Write a file")

    assert of_type(events, ToolApprovalRequested) == []
    assert (tmp_path / "out.txt").read_text() == "written"


async def test_the_callers_can_use_tool_decides_instead_of_the_approval_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def deny_all(tool_name: str, _input: dict[str, Any], _context: ToolPermissionContext) -> PermissionResultDeny:
        return PermissionResultDeny(message=f"{tool_name} is off limits.")

    # Without a permission mode, Claude Code writes files without asking can_use_tool.
    setup = ClientSetup(options={"can_use_tool": deny_all, "permission_mode": "default"})
    async with agent_session(
        monkeypatch, tmp_path, call_tools_then_reply(_write(tmp_path)), config=ASK_FIRST, setup=setup
    ) as session:
        events = await session.turn("Write a file")
        result = tool_results(session.api.requests[-1])["toolu_write"]

    assert of_type(events, ToolApprovalRequested) == []
    assert result.is_error and "Write is off limits." in result.content
    assert not (tmp_path / "out.txt").exists()


async def test_each_turns_tool_events_belong_to_its_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Claude Code asks for permission and calls MCP tools on its own, not in reply to a prompt.
    greet = tool_use("toolu_greet", "mcp__agent_kit__greet", {"name": "Claude"})
    respond = call_tools_then_reply(_write(tmp_path), greet)
    async with agent_session(monkeypatch, tmp_path, respond, config=ASK_FIRST) as session:
        first = await session.turn("First", approve=True)
        second = await session.turn("Second", approve=True)

    for events in (first, second):
        (message_id,) = {e.message_id for e in events if e.message_id is not None}
        tool_events = [
            e for e in events if isinstance(e.event, (ToolRequested, ToolApprovalRequested, ToolStartEvent, ToolEndEvent))
        ]
        assert {type(e.event) for e in tool_events} == {ToolRequested, ToolApprovalRequested, ToolStartEvent, ToolEndEvent}
        assert {(e.turn_id, e.message_id) for e in tool_events} == {(events[0].turn_id, message_id)}
    assert first[0].turn_id != second[0].turn_id
