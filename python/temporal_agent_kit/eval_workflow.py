"""Declare Workflow-backed eval suites and run their evaluators."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta
from typing import Literal, TypeVar, overload

from pydantic import BaseModel
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    is_cancelled_exception,
)

from temporal_agent_kit.agent_events import AgentEvents
from temporal_agent_kit.agent_protocol import AgentEvent
from temporal_agent_kit.evaluators import (
    _EVALUATOR_ATTR,
    EvaluatorInput,
    EvaluatorResult,
    _EvaluatorDefinition,
)
from temporal_agent_kit.trajectory_activities import AgentTurn, TrajectoryActivities

__all__ = ["EvalSuiteResult", "eval_suite_defn"]

_RETRY = RetryPolicy(maximum_attempts=3)
_SUITE_ATTR = "__agent_eval_suite__"


class EvalSuiteResult(BaseModel):
    """Outcomes keyed by evaluator name for the requested agent turn."""

    schema_version: Literal[1] = 1
    source: AgentTurn
    results: dict[str, EvaluatorResult]


_Suite = TypeVar("_Suite", bound=type)


@overload
def eval_suite_defn(cls: _Suite, /) -> _Suite: ...


@overload
def eval_suite_defn(*, name: str | None = None) -> Callable[[_Suite], _Suite]: ...


def eval_suite_defn(
    cls: _Suite | None = None, /, *, name: str | None = None
) -> _Suite | Callable[[_Suite], _Suite]:
    """Declare an eval suite Workflow from a class that lists its ``evaluators``.

    Register the class in the worker's workflows with ``AgentKitPlugin`` on the client.
    Start its generated ``run`` with an ``AgentTurn`` to capture once and run the evaluators
    concurrently. The Workflow type defaults to the class name.
    """

    def decorate(suite: _Suite) -> _Suite:
        if hasattr(suite, "run"):
            raise TypeError(f"{suite.__name__} defines run, which @agent.eval_suite_defn provides")
        evaluators = vars(suite).get("evaluators")
        if (
            not isinstance(evaluators, Sequence)
            or isinstance(evaluators, (str, bytes))
            or not evaluators
        ):
            raise TypeError(
                f"@agent.eval_suite_defn requires a non-empty {suite.__name__}.evaluators sequence"
            )
        definitions: dict[str, _EvaluatorDefinition] = {}
        for evaluator in evaluators:
            definition = getattr(evaluator, _EVALUATOR_ATTR, None)
            if not isinstance(definition, _EvaluatorDefinition):
                raise TypeError(
                    f"{suite.__name__}.evaluators accepts only @agent.eval_defn "
                    "and @agent.activity_eval_defn functions"
                )
            if definition.name in definitions:
                raise ValueError(f"duplicate evaluator name {definition.name!r} in {suite.__name__}")
            definitions[definition.name] = definition
        members = tuple(definitions.values())
        setattr(suite, _SUITE_ATTR, members)
        suite.run = _suite_run(suite, members)  # type: ignore[attr-defined]
        return workflow.defn(name=name)(suite)

    return decorate(cls) if cls is not None else decorate


def _suite_evaluators(suite: type) -> tuple[_EvaluatorDefinition, ...]:
    return vars(suite).get(_SUITE_ATTR, ())


def _suite_run(
    suite: type, definitions: tuple[_EvaluatorDefinition, ...]
) -> Callable[..., Awaitable[EvalSuiteResult]]:
    async def run(self: object, source: AgentTurn) -> EvalSuiteResult:
        events = await workflow.execute_activity_method(
            TrajectoryActivities.capture_trajectory,
            source,
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=_RETRY,
        )
        turn_id = events[-1].turn_id
        outcomes = await asyncio.gather(
            *(_run_evaluator(definition, events, turn_id) for definition in definitions)
        )
        return EvalSuiteResult(
            source=source,
            results={
                definition.name: outcome
                for definition, outcome in zip(definitions, outcomes, strict=True)
            },
        )

    run.__qualname__ = f"{suite.__qualname__}.run"
    run.__module__ = suite.__module__
    run.__annotations__ = {"source": AgentTurn, "return": EvalSuiteResult}
    return workflow.run(run)


async def _run_evaluator(
    definition: _EvaluatorDefinition, events: list[AgentEvent], turn_id: str
) -> EvaluatorResult:
    try:
        if definition.kind == "inline":
            # Nested payloads are mutable even when their event models are frozen.
            snapshot = AgentEvents(event.model_copy(deep=True) for event in events)
            return await definition.evaluate(EvaluatorInput(turn_id=turn_id, events=snapshot))
        return await workflow.execute_activity(
            definition.activity_type,
            args=[events, turn_id],
            result_type=EvaluatorResult,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=_RETRY,
        )
    except Exception as exc:
        if is_cancelled_exception(exc):
            raise
        return _execution_error(exc.cause if isinstance(exc, ActivityError) and exc.cause else exc)


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
