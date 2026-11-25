from __future__ import annotations

import pytest
from temporal_agent_kit import AgentEvents
from temporal_agent_kit.agent_protocol import (
    AgentClosed,
    AgentEvent,
    AgentStreamItem,
    AutoApprovalEvaluationError,
    CloseReason,
    MessageHandlerEnd,
    MessageHandlerError,
    ModelInteractionEnded,
    ModelInteractionStarted,
    ReplyDelta,
    SubagentMessageSent,
    TokenUsage,
    ToolErrorEvent,
    TurnStarted,
)

ROOT = "a1b2c3"
CHILD = "a1b2c3-d4e5f6"
GRANDCHILD = "a1b2c3-d4e5f6-aabbcc"


def observed(
    payload: AgentStreamItem,
    *,
    agent: str = ROOT,
    turn: str = "turn-1",
    message: str | None = "message-1",
    timestamp: float = 0,
) -> AgentEvent:
    return AgentEvent(
        agent_id=agent,
        turn_id=turn,
        turn_number=1,
        message_id=message,
        timestamp=timestamp,
        event=payload,
    )


def test_views_keep_native_order_and_scope_to_envelope_ids() -> None:
    child_delta = observed(ReplyDelta(text="child"), agent=CHILD, turn="child-turn", timestamp=20)
    root_marker = observed(
        SubagentMessageSent(
            subagent_id=CHILD,
            agent_key="researcher",
            workflow_id="child-wf",
            handler="chat",
            subagent_turn=1,
        ),
        timestamp=15,
    )
    grandchild_delta = observed(
        ReplyDelta(text="grandchild"), agent=GRANDCHILD, turn="grandchild-turn"
    )
    root_start = observed(TurnStarted(), turn="turn-2", message=None)
    root_delta = observed(ReplyDelta(text="root"), turn="turn-2", timestamp=10)
    supplied = [root_marker, child_delta, root_start, grandchild_delta, root_delta]
    events = AgentEvents(iter(supplied))

    assert events.all == supplied
    matching = events.of_type(ReplyDelta, TurnStarted)
    assert matching.all == [child_delta, root_start, grandchild_delta, root_delta]
    assert events.parent_events.all == [root_marker, root_start, root_delta]
    assert matching.parent_events.all == [root_start, root_delta]
    assert matching.for_agent(CHILD).all == [child_delta]
    assert matching.for_agent(CHILD).parent_events.all == []
    assert AgentEvents([child_delta, root_marker], parent_agent_id=ROOT).parent_events.all == [
        root_marker
    ]
    assert events.for_turn("turn-2").of_type(TurnStarted).all[0] is root_start
    assert events.of_type(AgentClosed).all == []
    assert events.for_agent(ROOT).all == [root_marker, root_start, root_delta]
    assert events.for_turn("turn-1").all == [root_marker]
    assert events.for_agent(CHILD).all == [child_delta]
    events.all.clear()
    assert events.all[0] is root_marker


def test_final_reply_requires_disambiguation_even_within_one_turn() -> None:
    delta = observed(ReplyDelta(text="draft"))
    child = observed(MessageHandlerEnd(output={"text": "child"}), agent=CHILD, turn="child-turn")
    joined = observed(MessageHandlerEnd(output={"text": "joined"}), message="joined")
    final = observed(MessageHandlerEnd(output={"text": "final"}))
    later = observed(MessageHandlerEnd(output={"text": "later"}), turn="turn-2", message="later")
    events = AgentEvents([delta, child, joined, final, later])

    assert events.replies == [child, joined, final, later]
    with pytest.raises(ValueError, match="multiple completed replies"):
        _ = events.for_agent(ROOT).for_turn("turn-1").reply_text
    assert events.for_agent(ROOT).for_turn("turn-1").for_message("message-1").reply_text == "final"
    assert events.for_agent(CHILD).reply_text == "child"
    assert events.for_turn("turn-2").reply_text == "later"


@pytest.mark.parametrize(
    ("output", "text"),
    [(None, None), ({"answer": 7}, None), ({"text": 7}, None), ({"text": ""}, "")],
)
def test_reply_text_comes_only_from_completed_output(
    output: dict[str, object] | None, text: str | None
) -> None:
    supplied = [observed(ReplyDelta(text="streamed draft"))]
    if output is not None:
        supplied.append(observed(MessageHandlerEnd(output=output)))

    assert AgentEvents(supplied).reply_text == text


def test_usage_aggregates_reported_fields_and_keeps_real_zero() -> None:
    events = AgentEvents(
        [
            observed(ModelInteractionStarted()),
            observed(
                ModelInteractionEnded(
                    usage=TokenUsage(input_tokens=5, output_tokens=0, total_tokens=8)
                )
            ),
            observed(ModelInteractionStarted(), agent=CHILD, turn="child-turn"),
            observed(
                ModelInteractionEnded(usage=TokenUsage(input_tokens=7, output_tokens=0)),
                agent=CHILD,
                turn="child-turn",
            ),
        ]
    )

    assert events.model_call_count == 2
    assert events.total_usage == TokenUsage(input_tokens=12, output_tokens=0)
    assert events.for_agent(ROOT).total_usage == TokenUsage(
        input_tokens=5, output_tokens=0, total_tokens=8
    )


def test_usage_is_unknown_for_unreported_or_unfinished_calls() -> None:
    start = observed(ModelInteractionStarted())
    completed = observed(ModelInteractionEnded(usage=TokenUsage(input_tokens=7)))
    unreported = observed(ModelInteractionEnded(), message="joined")
    other_start = observed(ModelInteractionStarted(), message="other")

    assert AgentEvents([]).total_usage == TokenUsage()
    assert AgentEvents([start, completed, unreported]).total_usage == TokenUsage()
    assert AgentEvents([other_start, start, completed]).total_usage == TokenUsage()
    assert AgentEvents([completed]).total_usage == TokenUsage(input_tokens=7)


def test_diagnostics_keep_trajectory_order_and_close_reason_is_agent_specific() -> None:
    events = AgentEvents(
        [
            observed(
                AutoApprovalEvaluationError(
                    tool_id="evaluation-tool",
                    tool_name="search",
                    evaluation_id="eval-1",
                    evaluator="approver",
                    message="approval eval failed",
                )
            ),
            observed(ToolErrorEvent(tool_id="tool-1", tool_name="search", message="tool failed")),
            observed(MessageHandlerError(message="handler failed")),
            observed(AgentClosed(reason=CloseReason.REQUESTED), turn="", message=None),
            observed(
                AgentClosed(reason=CloseReason.IDLE_TIMEOUT_EXPIRED),
                agent=CHILD,
                turn="",
                message=None,
            ),
        ]
    )

    assert events.errors == ["approval eval failed", "tool failed", "handler failed"]
    with pytest.raises(ValueError, match="multiple agent closes"):
        _ = events.close_reason
    assert events.for_agent(ROOT).close_reason is CloseReason.REQUESTED
    assert events.for_agent(CHILD).close_reason is CloseReason.IDLE_TIMEOUT_EXPIRED
    assert events.for_turn("turn-1").close_reason is None
