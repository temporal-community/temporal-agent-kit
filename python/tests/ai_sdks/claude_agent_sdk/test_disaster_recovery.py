"""Disaster recovery: the agent recovers when Claude Code or its Worker is lost, at each point in a turn.

Each disaster in ``DISASTERS`` hits an idle agent and a model call. The turn in progress is repeated on a
fresh Claude Code.
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
    ToolErrorEvent,
)
from tests.ai_sdks.claude_agent_sdk._helpers import (
    Session,
    agent_session,
    model_reply,
    of_type,
    reply_text,
    signal_claude,
    text,
    tool_results,
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
