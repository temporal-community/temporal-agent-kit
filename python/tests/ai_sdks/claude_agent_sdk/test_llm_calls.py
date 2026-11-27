"""Model calls: what a user sees stream from each of Claude's model calls, and what Claude Code asks the model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from temporal_agent_kit.agent_protocol import (
    ModelInteractionEnded,
    ModelInteractionStarted,
    ReplyDelta,
    ThoughtSummaryDelta,
    TokenUsage,
)
from tests.ai_sdks.claude_agent_sdk._helpers import (
    MODEL,
    ClientSetup,
    agent_session,
    model_reply,
    of_type,
    reply_text,
    text,
    thinking,
    tool_results,
    tool_use,
)


async def test_a_reply_streams_as_it_is_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with agent_session(monkeypatch, tmp_path, lambda _: model_reply(text("Hello there"))) as session:
        events = await session.turn("Hi")

    assert of_type(events, ReplyDelta) == [ReplyDelta(text="Hello there")]
    assert reply_text(events) == "Hello there"


async def test_thinking_streams_as_a_thought_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with agent_session(monkeypatch, tmp_path, lambda _: model_reply(thinking("Hmm", "sig-1"), text("Done"))) as session:
        events = await session.turn("Think")

    assert of_type(events, ThoughtSummaryDelta) == [
        ThoughtSummaryDelta(delta={"type": "thinking_delta", "text": "Hmm"}),
        ThoughtSummaryDelta(delta={"type": "signature_delta", "signature": "sig-1"}),
    ]


async def test_a_model_call_reports_its_model_and_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    usage = {"input_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5, "output_tokens": 1}
    async with agent_session(
        monkeypatch, tmp_path, lambda _: model_reply(text("Hi"), usage=usage, output_tokens=42)
    ) as session:
        events = await session.turn("Hi")

    assert [e for e in of_type(events, (ModelInteractionStarted, ModelInteractionEnded))] == [  # type: ignore[arg-type]
        ModelInteractionStarted(),
        ModelInteractionEnded(
            model=MODEL, usage=TokenUsage(input_tokens=115, output_tokens=42, cached_tokens=100, total_tokens=157)
        ),
    ]


async def test_each_model_call_in_a_turn_reports_its_own_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def respond(body: dict[str, Any]) -> bytes:
        if tool_results(body):
            return model_reply(text("Done"), model="second-model", usage={"input_tokens": 3, "output_tokens": 1}, output_tokens=2)
        return model_reply(tool_use("toolu_1", "mcp__agent_kit__greet", {"name": "Claude"}))

    async with agent_session(monkeypatch, tmp_path, respond) as session:
        events = await session.turn("Greet Claude")

    assert [(e.model, e.usage and e.usage.total_tokens) for e in of_type(events, ModelInteractionEnded)] == [
        (MODEL, 15),
        ("second-model", 5),
    ]


async def test_thinking_is_summarized_unless_the_caller_configures_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with agent_session(monkeypatch, tmp_path, lambda _: model_reply(text("Hi"))) as session:
        await session.turn("Hi")
        default_request = session.api.requests[-1]

    omitted = {"type": "adaptive", "display": "omitted"}
    async with agent_session(
        monkeypatch, tmp_path, lambda _: model_reply(text("Hi")), setup=ClientSetup(options={"thinking": omitted})
    ) as session:
        events = await session.turn("Hi")
        configured_request = session.api.requests[-1]

    assert default_request["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert configured_request["thinking"]["display"] != "summarized"
    assert of_type(events, ReplyDelta) == [ReplyDelta(text="Hi")]  # Streaming stays on
