"""Disaster recovery: the agent recovers when Claude Code or its Worker is lost, at each point in a turn.

Each disaster in ``DISASTERS`` hits an idle agent, a model call, a pending tool approval, and a running
tool. The turn in progress is repeated on a fresh Claude Code.
"""

from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from temporal_agent_kit.agent_protocol import (
    AgentEvent,
    ModelInteractionEnded,
    ModelInteractionStarted,
    ToolApprovalRequested,
    ToolApprovalResolved,
    ToolErrorEvent,
    ToolStartEvent,
)
from tests.ai_sdks.claude_agent_sdk._helpers import (
    ASK_FIRST,
    TOOL_CANCELLED,
    Session,
    agent_session,
    assert_tools_settled,
    model_reply,
    of_type,
    reply_text,
    signal_claude,
    text,
    tool_results,
    tool_use,
)


@dataclass(frozen=True)
class Disaster:
    name: str
    strike: Callable[[Session], Awaitable[None]]

    def __repr__(self) -> str:
        return self.name


async def _kill_claude(sig: signal.Signals) -> None:
    assert signal_claude(sig) >= 1


async def _lose_worker(session: Session, *, stop_claude: bool) -> None:
    await session.restart_worker(before_start=(lambda: signal_claude(signal.SIGTERM)) if stop_claude else None)


DISASTERS = [
    pytest.param(Disaster("claude_soft_crash", lambda _: _kill_claude(signal.SIGTERM)), id="claude_soft_crash"),
    pytest.param(Disaster("claude_hard_crash", lambda _: _kill_claude(signal.SIGKILL)), id="claude_hard_crash"),
    pytest.param(Disaster("worker_soft_crash", lambda s: _lose_worker(s, stop_claude=True)), id="worker_soft_crash"),
    pytest.param(Disaster("worker_hard_crash", lambda s: _lose_worker(s, stop_claude=False)), id="worker_hard_crash"),
]

WRITE = tool_use("toolu_write", "Write", {"file_path": "out.txt", "content": "written"})
GREET = tool_use("toolu_greet", "mcp__agent_kit__greet", {"name": "Claude"})
SLEEP = tool_use("toolu_sleep", "Bash", {"command": "sleep 20 && touch slept.txt"})
WAIT = tool_use("toolu_wait", "mcp__agent_kit__wait_a_while")


class _Model:
    """A mock model that answers each new prompt with the next of ``calls``, and replies once tools return.

    Holds the model calls numbered in ``hold`` until the test releases them.
    """

    def __init__(self, *calls: tuple[dict[str, Any], ...], hold: tuple[int, ...] = ()) -> None:
        self._calls = calls
        self._prompts = 0
        self._requests = 0
        self.reached = {n: asyncio.Event() for n in hold}
        self.release = {n: asyncio.Event() for n in hold}

    async def __call__(self, body: dict[str, Any]) -> bytes:
        self._requests += 1
        if (n := self._requests) in self.reached:
            self.reached[n].set()
            await self.release[n].wait()
        results = tool_results(body)
        if results:
            return model_reply(text(json.dumps({k: v.is_error for k, v in sorted(results.items())})))
        self._prompts += 1
        calls = self._calls[min(self._prompts, len(self._calls)) - 1] if self._calls else ()
        if not calls:
            return model_reply(text("Recovered."))
        return model_reply(*({**c, "id": f"{c['id']}_{self._prompts}"} for c in calls))

    def release_all(self) -> None:
        for event in self.release.values():
            event.set()


def _assert_model_calls_ended(events: list[AgentEvent]) -> None:
    """Each model call that started ended, even one the disaster cut short. Losing Claude Code may end one more."""
    assert len(of_type(events, ModelInteractionEnded)) >= len(of_type(events, ModelInteractionStarted))


@pytest.mark.parametrize("disaster", DISASTERS)
async def test_a_disaster_while_idle_between_turns(
    disaster: Disaster, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with agent_session(monkeypatch, tmp_path) as session:
        await session.turn("First")
        await disaster.strike(session)
        events = await session.turn("Second")

    assert reply_text(events) == "ECHO:Second"
    assert of_type(events, ToolErrorEvent) == []
    _assert_model_calls_ended(events)


@pytest.mark.parametrize("disaster", DISASTERS)
async def test_a_disaster_during_a_model_call_repeats_the_turn(
    disaster: Disaster, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _Model(hold=(1,))
    async with agent_session(monkeypatch, tmp_path, model) as session:
        turn = asyncio.create_task(session.collect(session.start_turn("Hi")))
        await model.reached[1].wait()
        await disaster.strike(session)
        model.release_all()
        events = await turn

    assert reply_text(events) == "Recovered."
    _assert_model_calls_ended(events)


@pytest.mark.parametrize("pending", [pytest.param(WRITE, id="built_in"), pytest.param(GREET, id="agent_kit")])
@pytest.mark.parametrize("disaster", DISASTERS)
async def test_a_disaster_during_a_pending_tool_approval_cancels_it(
    disaster: Disaster, pending: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool_id = f"{pending['id']}_1"
    model = _Model((pending,), ())
    async with agent_session(monkeypatch, tmp_path, model, config=ASK_FIRST) as session:

        async def strike_when_asked(event: AgentEvent) -> None:
            if isinstance(event.event, ToolApprovalRequested) and event.event.tool_id == tool_id:
                await disaster.strike(session)

        events = await session.collect(session.start_turn("Use the tool"), on_event=strike_when_asked)

    assert reply_text(events) == "Recovered."
    assert [(e.tool_id, e.approved, e.reason) for e in of_type(events, ToolApprovalResolved)] == [
        (tool_id, False, TOOL_CANCELLED)
    ]
    assert of_type(events, ToolStartEvent) == []
    assert not (tmp_path / "out.txt").exists()
    assert_tools_settled(events)
    _assert_model_calls_ended(events)


@pytest.mark.parametrize("running", [pytest.param(SLEEP, id="built_in"), pytest.param(WAIT, id="agent_kit")])
@pytest.mark.parametrize("disaster", DISASTERS)
async def test_a_disaster_during_a_running_tool_fails_it(
    disaster: Disaster, running: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool_id = f"{running['id']}_1"
    model = _Model((running,), ())
    async with agent_session(monkeypatch, tmp_path, model) as session:
        struck = False

        async def strike_once_it_runs(event: AgentEvent) -> None:
            nonlocal struck
            if isinstance(event.event, ToolStartEvent) and event.event.tool_id == tool_id and not struck:
                struck = True
                await asyncio.sleep(0.5)  # Let it get going
                await disaster.strike(session)

        events = await session.collect(session.start_turn("Use the tool"), on_event=strike_once_it_runs)

    assert struck
    assert reply_text(events) == "Recovered."
    assert [e.tool_id for e in of_type(events, ToolErrorEvent)] == [tool_id]
    assert not (tmp_path / "slept.txt").exists()
    assert_tools_settled(events)
    _assert_model_calls_ended(events)
