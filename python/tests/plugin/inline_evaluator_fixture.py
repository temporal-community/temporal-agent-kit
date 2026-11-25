"""Inline evaluators imported by the Workflow sandbox during integration tests."""

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporal_agent_kit import EvaluatorInput, EvaluatorResult, agent
    from temporal_agent_kit.agent_protocol import MessageHandlerEnd


@agent.eval_defn
async def clear_reply_text(evaluation: EvaluatorInput) -> EvaluatorResult:
    replies = evaluation.events.replies
    for item in replies:
        if isinstance(item.event, MessageHandlerEnd):
            item.event.output["text"] = ""
    return EvaluatorResult(status="completed", value=len(replies))


@agent.eval_defn
async def inline_reply_length(evaluation: EvaluatorInput) -> EvaluatorResult:
    return EvaluatorResult(status="completed", value=len(evaluation.turn_events.reply_text or ""))


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


# Register on the worker so the plan includes it, then omit it from the Workflow sandbox.
if not workflow.unsafe.in_sandbox():
    @agent.eval_defn
    async def unavailable_inline_evaluator(evaluation: EvaluatorInput) -> EvaluatorResult:
        return EvaluatorResult(status="completed", value=42)
