"""Inline and Activity evals for OpenAI Hello."""

from __future__ import annotations

from temporal_agent_kit import AgentEvents, agent
from temporal_agent_kit.agent_protocol import MessageHandlerEnd
from temporal_agent_kit.evaluators import EvaluatorResult


@agent.eval_defn
async def has_reply(events: AgentEvents) -> EvaluatorResult:
    """Score whether the final captured turn produced a nonblank reply."""
    turn = events.for_turn(events.all[-1].turn_id)
    has_text = any(
        isinstance(text, str) and text.strip()
        for item in turn.replies
        if isinstance(reply := item.event, MessageHandlerEnd)
        if (text := reply.output.get("text")) is not None
    )
    return EvaluatorResult(
        status="completed",
        value=1 if has_text else 0,
        value_range=(0, 1),
    )


@agent.activity_eval_defn
async def count_failed_tools(events: AgentEvents) -> EvaluatorResult:
    """Count failed tool calls in the final captured turn."""
    turn = events.for_turn(events.all[-1].turn_id)
    return EvaluatorResult(
        status="completed",
        value=sum(call.error is not None for call in turn.tool_calls),
    )
