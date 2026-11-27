"""Steering: a message the user sends mid-turn reaches Claude with the next tool result, or is dropped at the turn's end."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import HookMatcher

from temporal_agent_kit.agent_protocol import (
    SteeringClaimed,
    SteeringDropped,
    SteeringEnqueued,
)
from tests.ai_sdks.claude_agent_sdk._helpers import (
    ClientSetup,
    Responder,
    Session,
    agent_session,
    call_tools_then_reply,
    model_reply,
    of_type,
    text,
    tool_use,
)

STEERING = "Make it short."

BASH = tool_use("toolu_bash", "Bash", {"command": "printf built-in | tee out.txt"})
GREET = tool_use("toolu_greet", "mcp__agent_kit__greet", {"name": "Claude"})
RUN_CODE = tool_use("toolu_code", "mcp__agent_kit__run_code", {"script": "await greet(name='Code')"})
FAILING_READ = tool_use("toolu_read", "Read", {"file_path": "/no/such/file.txt"})
EXPLODE = tool_use("toolu_explode", "mcp__agent_kit__explode")


class _HeldModel:
    """A mock model that holds its first call until the test releases it."""

    def __init__(self, respond: Responder) -> None:
        self._respond = respond
        self.reached = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, body: dict[str, Any]) -> bytes:
        if not self.reached.is_set():
            self.reached.set()
            await self.release.wait()
        result = self._respond(body)
        return await result if asyncio.iscoroutine(result) else result  # type: ignore[return-value]


async def _steered_turn(session: Session, model: _HeldModel) -> list[Any]:
    """Run a turn, steering it while the model works on its first call."""
    turn = asyncio.create_task(session.collect(session.start_turn("Do the work")))
    await model.reached.wait()
    await session.steer(STEERING)
    model.release.set()
    return await turn


@pytest.mark.parametrize(
    "calls",
    [
        pytest.param((BASH,), id="built_in"),
        pytest.param((GREET,), id="agent_kit"),
        pytest.param((RUN_CODE,), id="code_mode"),
        pytest.param((BASH, GREET, RUN_CODE), id="all_parallel"),
        pytest.param((FAILING_READ,), id="failed_built_in"),
        pytest.param((EXPLODE,), id="failed_agent_kit"),
    ],
)
async def test_the_first_finished_tool_brings_the_steering_to_claude(
    calls: tuple[dict[str, Any], ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _HeldModel(call_tools_then_reply(*calls))
    async with agent_session(monkeypatch, tmp_path, model) as session:
        events = await _steered_turn(session, model)
        after_tools = json.dumps(session.api.requests[-1]["messages"])

    (enqueued,) = of_type(events, SteeringEnqueued)
    (claimed,) = of_type(events, SteeringClaimed)  # Parallel tools find nothing left to claim
    assert claimed.steering_ids == [enqueued.steering_id]
    assert claimed.tool_id in {c["id"] for c in calls}
    assert of_type(events, SteeringDropped) == []
    assert STEERING in after_tools


async def test_steering_in_a_turn_without_tools_is_dropped_at_its_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _HeldModel(lambda _: model_reply(text("A poem.")))
    async with agent_session(monkeypatch, tmp_path, model) as session:
        events = await _steered_turn(session, model)

    (enqueued,) = of_type(events, SteeringEnqueued)
    (dropped,) = of_type(events, SteeringDropped)
    assert dropped.steering_ids == [enqueued.steering_id]
    assert of_type(events, SteeringClaimed) == []


async def test_the_callers_tool_hooks_still_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []

    def record(label: str) -> Any:
        async def hook(hook_input: dict[str, Any], _tool_use_id: str | None, _context: Any) -> dict[str, Any]:
            seen.append(f"{label}:{hook_input['tool_name']}")
            return {}

        return hook

    hooks = {
        "PreToolUse": [HookMatcher(matcher="Bash", hooks=[record("pre")])],
        "PostToolUse": [HookMatcher(hooks=[record("post")])],
    }
    model = _HeldModel(call_tools_then_reply(BASH))
    async with agent_session(monkeypatch, tmp_path, model, setup=ClientSetup(options={"hooks": hooks})) as session:
        events = await _steered_turn(session, model)

    assert seen == ["pre:Bash", "post:Bash"]
    assert len(of_type(events, SteeringClaimed)) == 1  # Agent Kit's own hooks ran beside them
