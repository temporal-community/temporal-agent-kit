"""Capture a turn and run the agent's registered evaluators."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    is_cancelled_exception,
)

from temporal_agent_kit.agent_events import AgentEvents
from temporal_agent_kit.agent_protocol import AgentEvent
from temporal_agent_kit.evaluators import (
    EvaluatorInput,
    EvaluatorResult,
    _EvaluatorSpec,
    _load_inline_evaluator,
)
from temporal_agent_kit.trajectory_activities import AgentTurn, TrajectoryActivities

__all__ = ["RunEvaluatorsResult", "RunEvaluatorsWorkflow"]

_RETRY = RetryPolicy(maximum_attempts=3)


class RunEvaluatorsResult(BaseModel):
    """Outcomes for the evaluators selected when the run started."""

    schema_version: Literal[1] = 1
    agent: str
    source: AgentTurn
    results: dict[str, EvaluatorResult]


class _EvalPlan(BaseModel):
    agent: str
    evaluators: list[_EvaluatorSpec]


class _EvalRegistry:
    def __init__(
        self,
        client: Client,
        default: list[_EvaluatorSpec],
        by_agent: dict[str, list[_EvaluatorSpec]],
    ) -> None:
        self._client = client
        self._default = default
        self._by_agent = by_agent

    @activity.defn(name="agentkit.eval.plan")
    async def plan(self, source: AgentTurn) -> _EvalPlan:
        handle = self._client.get_workflow_handle(source.workflow_id, run_id=source.run_id)
        agent = (await handle.describe()).workflow_type
        if self._by_agent and agent not in self._by_agent:
            registered = ", ".join(repr(name) for name in sorted(self._by_agent))
            raise ApplicationError(
                f"No evaluators registered for source Workflow type {agent!r}. "
                f"Registered Workflow types are {registered}.",
                type="EvaluatorsNotConfigured",
                non_retryable=True,
            )
        return _EvalPlan(agent=agent, evaluators=self._by_agent.get(agent, self._default))


@workflow.defn
class RunEvaluatorsWorkflow:
    @workflow.run
    async def run(self, source: AgentTurn) -> RunEvaluatorsResult:
        plan = await workflow.execute_activity_method(
            _EvalRegistry.plan,
            source,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_RETRY,
        )
        results: dict[str, EvaluatorResult] = {}
        if plan.evaluators:
            events = await workflow.execute_activity_method(
                TrajectoryActivities.capture_trajectory,
                source,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=_RETRY,
            )
            outcomes = await asyncio.gather(
                *(
                    _run_inline_evaluator(spec, events, source.turn_id)
                    if spec.kind == "inline"
                    else _run_activity_evaluator(spec, events, source.turn_id)
                    for spec in plan.evaluators
                )
            )
            results = {
                spec.name: outcome for spec, outcome in zip(plan.evaluators, outcomes, strict=True)
            }
        return RunEvaluatorsResult(agent=plan.agent, source=source, results=results)


async def _run_inline_evaluator(
    spec: _EvaluatorSpec, events: list[AgentEvent], turn_id: str
) -> EvaluatorResult:
    try:
        definition = _load_inline_evaluator(spec)
        # Nested payloads are mutable even when their event models are frozen.
        snapshot = AgentEvents(event.model_copy(deep=True) for event in events)
        return await definition.evaluate(EvaluatorInput(turn_id=turn_id, events=snapshot))
    except Exception as exc:
        if is_cancelled_exception(exc):
            raise
        return _execution_error(exc)


async def _run_activity_evaluator(
    spec: _EvaluatorSpec, events: list[AgentEvent], turn_id: str
) -> EvaluatorResult:
    try:
        return await workflow.execute_activity(
            spec.activity_type,
            args=[events, turn_id],
            result_type=EvaluatorResult,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=_RETRY,
        )
    except ActivityError as exc:
        if is_cancelled_exception(exc):
            raise
        return _execution_error(exc.cause or exc)


def _execution_error(exc: BaseException) -> EvaluatorResult:
    error_type = type(exc).__name__
    if isinstance(exc, ApplicationError):
        error_type = exc.type or error_type
        message = exc.message.strip()
    else:
        message = str(exc).strip()
    return EvaluatorResult(
        status="error",
        error_kind="execution",
        reasoning=f"{error_type}: {message}" if message else error_type,
    )
