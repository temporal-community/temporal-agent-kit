"""Inline evaluator imported by the Workflow sandbox during integration tests."""

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporal_agent_kit import AgentEvents, agent
    from temporal_agent_kit.evaluators import EvaluatorResult


@agent.eval_defn()
async def invalid_inline_result(events: AgentEvents) -> EvaluatorResult:
    result = EvaluatorResult(status="completed", value=42)
    result.status = "skipped"
    return result


# Register on the worker so the plan includes it, then omit it from the Workflow sandbox.
if not workflow.unsafe.in_sandbox():
    @agent.eval_defn
    async def unavailable_inline_evaluator(events: AgentEvents) -> EvaluatorResult:
        return EvaluatorResult(status="completed", value=42)
