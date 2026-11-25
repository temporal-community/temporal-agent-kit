"""An eval suite the integration tests run, imported by the Workflow sandbox like any workflow."""

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from examples.openai_hello.evals import (
        count_failed_tools,
        reply_is_brief,
        total_input_tokens,
    )
    from temporal_agent_kit import agent
    from temporal_agent_kit.agent_protocol import MessageHandlerEnd, ToolErrorEvent
    from temporal_agent_kit import ScorerResult, Trajectory


@agent.eval_scorer_defn
async def clear_reply_text(trajectory: Trajectory) -> ScorerResult:
    replies = trajectory.events.replies
    for item in replies:
        if isinstance(item.event, MessageHandlerEnd):
            item.event.output["text"] = ""
    return ScorerResult(status="completed", value=len(replies))


@agent.eval_scorer_defn
async def crashing_inline_scorer(trajectory: Trajectory) -> ScorerResult:
    raise KeyError("text")


@agent.eval_scorer_defn
async def reported_inline_error(trajectory: Trajectory) -> ScorerResult:
    return ScorerResult(status="error", reasoning="Could not score inline.")


@agent.activity_eval_scorer_defn
async def reported_activity_error(trajectory: Trajectory) -> ScorerResult:
    return ScorerResult(status="error", reasoning="Could not score in the Activity.")


@agent.eval_scorer_defn
async def invalid_inline_result(trajectory: Trajectory) -> ScorerResult:
    result = ScorerResult(status="completed", value=42)
    result.status = "skipped"
    return result


@agent.activity_eval_scorer_defn
async def prior_failed_tools(trajectory: Trajectory) -> ScorerResult:
    if len(trajectory.events.replies) == len(trajectory.last_turn.replies):
        return ScorerResult(status="skipped", reasoning="No earlier replies.")
    all_failures = len(trajectory.events.of_type(ToolErrorEvent).all)
    turn_failures = len(trajectory.last_turn.of_type(ToolErrorEvent).all)
    return ScorerResult(status="completed", value=all_failures - turn_failures)


@agent.activity_eval_scorer_defn
async def unavailable_judge(trajectory: Trajectory) -> ScorerResult:
    raise ApplicationError("Judge unavailable.", type="JudgeUnavailable", non_retryable=True)


def _replying_agents(trajectory: Trajectory) -> int:
    return len({item.agent_id for item in trajectory.last_turn.of_type(MessageHandlerEnd).all})


@agent.eval_scorer_defn(include_subagents=True)
async def tree_replying_agents(trajectory: Trajectory) -> ScorerResult:
    return ScorerResult(status="completed", value=_replying_agents(trajectory))


@agent.eval_scorer_defn
async def root_replying_agents(trajectory: Trajectory) -> ScorerResult:
    return ScorerResult(status="completed", value=_replying_agents(trajectory))


@agent.eval_suite_defn
class ScriptedAgentEvals:
    scorers = [
        clear_reply_text,
        count_failed_tools,
        reply_is_brief,
        total_input_tokens,
        prior_failed_tools,
        invalid_inline_result,
        unavailable_judge,
        crashing_inline_scorer,
        reported_inline_error,
        reported_activity_error,
        tree_replying_agents,
        root_replying_agents,
    ]
