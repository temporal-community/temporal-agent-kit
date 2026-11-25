"""An eval suite imported by the Workflow sandbox during integration tests."""

from temporal_agent_kit import EvaluatorInput, EvaluatorResult, agent
from temporal_agent_kit.agent_protocol import MessageHandlerEnd, ToolErrorEvent
from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from examples.openai_hello.evaluators import (
        count_failed_tools,
        reply_is_brief,
        total_input_tokens,
    )


@agent.eval_defn
async def clear_reply_text(evaluation: EvaluatorInput) -> EvaluatorResult:
    replies = evaluation.events.replies
    for item in replies:
        if isinstance(item.event, MessageHandlerEnd):
            item.event.output["text"] = ""
    return EvaluatorResult(status="completed", value=len(replies))


@agent.eval_defn
async def inline_reply_length(evaluation: EvaluatorInput) -> EvaluatorResult:
    assert workflow.unsafe.in_sandbox()
    reply = evaluation.turn_events.source_events.reply_text
    return EvaluatorResult(status="completed", value=len(reply or ""))


@agent.eval_defn()
async def invalid_inline_result(evaluation: EvaluatorInput) -> EvaluatorResult:
    result = EvaluatorResult(status="completed", value=42)
    result.status = "skipped"
    return result


@agent.eval_defn
async def crashing_inline_evaluator(evaluation: EvaluatorInput) -> EvaluatorResult:
    raise KeyError("text")


@agent.eval_defn
async def reported_inline_error(evaluation: EvaluatorInput) -> EvaluatorResult:
    return EvaluatorResult(status="error", reasoning="Could not score inline.")


@agent.activity_eval_defn()
async def prior_failed_tools(evaluation: EvaluatorInput) -> EvaluatorResult:
    events = evaluation.events
    turn = evaluation.turn_events
    if len(events.replies) == len(turn.replies):
        return EvaluatorResult(status="skipped", reasoning="No earlier replies.")
    all_failures = len(events.of_type(ToolErrorEvent).all)
    target_failures = len(turn.of_type(ToolErrorEvent).all)
    return EvaluatorResult(status="completed", value=all_failures - target_failures)


@agent.activity_eval_defn
async def unavailable_judge(evaluation: EvaluatorInput) -> EvaluatorResult:
    raise ApplicationError("Judge unavailable.", type="JudgeUnavailable", non_retryable=True)


@agent.activity_eval_defn
async def reported_activity_error(evaluation: EvaluatorInput) -> EvaluatorResult:
    return EvaluatorResult(status="error", reasoning="Could not score in the Activity.")


@agent.eval_suite_defn(name="ScriptedAgentEvals")
class ScriptedAgentEvals:
    evaluators = (
        clear_reply_text,
        inline_reply_length,
        count_failed_tools,
        reply_is_brief,
        total_input_tokens,
        prior_failed_tools,
        invalid_inline_result,
        crashing_inline_evaluator,
        reported_inline_error,
        unavailable_judge,
        reported_activity_error,
    )
