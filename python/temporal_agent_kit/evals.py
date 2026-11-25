"""Eval scorers, and the eval suite workflows that run them over an agent session's trajectory."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, TypeVar, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, is_cancelled_exception

from temporal_agent_kit.agent_events import AgentEvents
from temporal_agent_kit.agent_protocol import TurnStarted
from temporal_agent_kit.trajectory_activities import (
    CAPTURE_TRAJECTORY_ACTIVITY,
    AgentTurn,
    CapturedTurn,
)

__all__ = [
    "AgentTurn",
    "EvalSuiteResult",
    "ScorerResult",
    "Trajectory",
    "activity_eval_scorer_defn",
    "eval_scorer_defn",
    "eval_suite_defn",
]

_RETRY = RetryPolicy(maximum_attempts=3)
_SUITE_ATTR = "__agent_eval_suite__"
_SCORER_ATTR = "__agent_eval_scorer__"


class ScorerResult(BaseModel):
    """The outcome of one scorer. A numeric value can have an inclusive range.

    ``error_kind`` says where an error came from: ``reported`` when the scorer returned it,
    ``execution`` when the scorer raised, returned an invalid result, or could not run.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    status: Literal["completed", "skipped", "error"]
    value: float | str | None = None
    value_range: tuple[float, float] | None = None
    reasoning: str | None = None
    error_kind: Literal["reported", "execution"] | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> ScorerResult:
        if self.status == "completed" and self.value is None:
            raise ValueError("completed outcomes require a value")
        if self.status != "completed" and (
            self.value is not None or not (self.reasoning or "").strip()
        ):
            raise ValueError("skipped and error outcomes require reasoning and no score")
        if self.status != "error" and self.error_kind is not None:
            raise ValueError("error_kind is only allowed for error outcomes")
        if self.value_range is None:
            return self
        if self.value is None or isinstance(self.value, str):
            raise ValueError("value_range requires a numeric value")
        lower, upper = self.value_range
        if lower > upper:
            raise ValueError("value_range lower bound must be less than or equal to its upper bound")
        if not lower <= self.value <= upper:
            raise ValueError("value must be within value_range")
        return self


@dataclass(frozen=True)
class Trajectory:
    """An agent session from its start through the end of one completed turn.

    A scorer judges the whole trajectory, so it can score the session on anything that happened
    in any turn up to ``source.turn_number``.

    ``events`` holds every captured event, in merged order. ``last_turn`` narrows that to the
    final turn: the root agent's events for it plus, for a scorer that includes subagents,
    every subagent event emitted while it ran.
    """

    source: AgentTurn
    root_agent_id: str
    events: AgentEvents
    last_turn: AgentEvents

    @classmethod
    def from_capture(cls, capture: CapturedTurn) -> Trajectory:
        root, number = capture.root_agent_id, capture.source.turn_number
        events = capture.events
        start = next(
            (
                index
                for index, item in enumerate(events)
                if item.agent_id == root
                and item.turn_number == number
                and isinstance(item.event, TurnStarted)
            ),
            None,
        )
        if start is None:
            raise ValueError(f"the capture does not contain the start of turn {number}")
        # The capture ends at the root's TurnEnded for this turn. Root events for the turn can
        # precede its TurnStarted (a queued message is accepted during the previous turn).
        last_turn = [
            item
            for index, item in enumerate(events)
            if (item.turn_number == number if item.agent_id == root else index > start)
        ]
        return cls(
            source=capture.source,
            root_agent_id=root,
            events=AgentEvents(events),
            last_turn=AgentEvents(last_turn),
        )


Scorer = Callable[[Trajectory], Awaitable[ScorerResult]]
_E = TypeVar("_E", bound=Scorer)
_ScorerKind = Literal["inline", "activity"]


@dataclass(frozen=True)
class _ScorerDefinition:
    name: str
    activity_type: str
    kind: _ScorerKind
    include_subagents: bool
    fn: Scorer

    def scope(self, capture: CapturedTurn) -> CapturedTurn:
        return capture if self.include_subagents else capture.root_only()

    def bind_activity(self) -> Callable[[CapturedTurn], Awaitable[ScorerResult]]:
        @activity.defn(name=self.activity_type)
        async def run(capture: CapturedTurn) -> ScorerResult:
            return await self.score(capture)

        return run

    async def score(self, capture: CapturedTurn) -> ScorerResult:
        result = await self.fn(Trajectory.from_capture(capture))
        try:
            # Validate a fresh snapshot because the author may have mutated a result model.
            outcome = ScorerResult.model_validate(
                result.model_dump() if isinstance(result, ScorerResult) else result
            )
        except ValidationError as exc:
            reason = exc.errors(include_input=False, include_url=False)[0]["msg"]
            raise ApplicationError(
                reason, type="InvalidScorerResult", non_retryable=True
            ) from None
        if outcome.status == "error":
            outcome.error_kind = "reported"
        return outcome


@overload
def eval_scorer_defn(fn: _E, /) -> _E: ...
@overload
def eval_scorer_defn(*, include_subagents: bool = False) -> Callable[[_E], _E]: ...
def eval_scorer_defn(
    fn: _E | None = None, /, *, include_subagents: bool = False
) -> _E | Callable[[_E], _E]:
    """Define a scorer that runs as Workflow code inside its eval suite's workflow.

    It follows Workflow rules: it is deterministic, and does any I/O, such as model calls,
    through Activities it starts. Use it for checks computed from the trajectory alone, or for
    multi-step scorers where each step should be retried and recorded separately. With
    ``include_subagents``, the trajectory includes every subagent's events; otherwise it holds
    only the root agent's.
    """
    return _define_scorer(fn, kind="inline", include_subagents=include_subagents)


@overload
def activity_eval_scorer_defn(fn: _E, /) -> _E: ...
@overload
def activity_eval_scorer_defn(*, include_subagents: bool = False) -> Callable[[_E], _E]: ...
def activity_eval_scorer_defn(
    fn: _E | None = None, /, *, include_subagents: bool = False
) -> _E | Callable[[_E], _E]:
    """Define a scorer that runs whole in one Activity, retried as a unit.

    Use it to call a model or a service directly, as a model judge does. With ``include_subagents``, the trajectory includes every subagent's events; otherwise it
    holds only the root agent's.
    """
    return _define_scorer(fn, kind="activity", include_subagents=include_subagents)


def _define_scorer(
    fn: _E | None, *, kind: _ScorerKind, include_subagents: bool
) -> _E | Callable[[_E], _E]:
    decorator = (
        "@agent.eval_scorer_defn" if kind == "inline" else "@agent.activity_eval_scorer_defn"
    )

    def decorate(scorer: _E) -> _E:
        if hasattr(scorer, _SCORER_ATTR):
            raise TypeError(f"{decorator} requires a function with only one scorer decorator")
        parameters = tuple(inspect.signature(scorer).parameters.values())
        if (
            not inspect.iscoroutinefunction(scorer)
            or len(parameters) != 1
            or parameters[0].kind
            not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ):
            raise TypeError(f"{decorator} requires an async function taking one Trajectory")
        hints = get_type_hints(scorer)
        if (
            hints.get(parameters[0].name) is not Trajectory
            or hints.get("return") is not ScorerResult
        ):
            raise TypeError(f"{decorator} requires (Trajectory) -> ScorerResult")
        definition = _ScorerDefinition(
            name=scorer.__name__,
            activity_type=f"agentkit.eval.{scorer.__module__}.{scorer.__qualname__}",
            kind=kind,
            include_subagents=include_subagents,
            fn=scorer,
        )
        vars(scorer)[_SCORER_ATTR] = definition
        return scorer

    return decorate(fn) if fn is not None else decorate


class EvalSuiteResult(BaseModel):
    """Each scorer's outcome for a session through one turn, keyed by scorer function name."""

    schema_version: Literal[1] = 1
    source: AgentTurn
    results: dict[str, ScorerResult]


_Suite = TypeVar("_Suite", bound=type)


@overload
def eval_suite_defn(cls: _Suite, /) -> _Suite: ...
@overload
def eval_suite_defn(*, name: str | None = None) -> Callable[[_Suite], _Suite]: ...
def eval_suite_defn(
    cls: _Suite | None = None, /, *, name: str | None = None
) -> _Suite | Callable[[_Suite], _Suite]:
    """Declare an eval suite: a workflow that scores any agent's session through a completed turn.

    List the suite's scorers in a ``scorers`` class attribute::

        @agent.eval_suite_defn(name="HelloEvals")
        class HelloEvals:
            scorers = [reply_is_brief, request_type]

    Register the suite on a worker like any workflow, with ``AgentKitPlugin`` on the client.
    Start it with an :class:`AgentTurn`. It captures the turn once, runs every scorer
    concurrently, and returns an :class:`EvalSuiteResult`. A turn that has not completed fails
    the run before any scorer starts. A scorer that raises gets an ``error`` result
    without affecting the others.

    ``name`` is the workflow type name and defaults to the class name.
    """

    def decorate(c: _Suite) -> _Suite:
        if "run" in vars(c):
            raise TypeError(
                f"{c.__name__!r} defines `run`, but @agent.eval_suite_defn provides it"
            )
        definitions = _suite_scorers(c)
        setattr(c, _SUITE_ATTR, definitions)
        c.run = _generated_suite_run(c, definitions)  # type: ignore[attr-defined]
        return workflow.defn(name=name)(c)

    return decorate(cls) if cls is not None else decorate


def _suite_scorers(cls: type) -> tuple[_ScorerDefinition, ...]:
    scorers = vars(cls).get("scorers")
    if not isinstance(scorers, Sequence) or isinstance(scorers, str) or not scorers:
        raise TypeError(
            f"@agent.eval_suite_defn requires {cls.__name__}.scorers to list scorers"
        )
    definitions: list[_ScorerDefinition] = []
    for scorer in scorers:
        definition = getattr(scorer, _SCORER_ATTR, None)
        if not isinstance(definition, _ScorerDefinition):
            raise TypeError(
                f"{cls.__name__}.scorers accepts only @agent.eval_scorer_defn and "
                "@agent.activity_eval_scorer_defn functions"
            )
        if any(existing.name == definition.name for existing in definitions):
            raise ValueError(f"duplicate scorer name {definition.name!r} in {cls.__name__}")
        definitions.append(definition)
    return tuple(definitions)


def suite_activities(cls: type) -> list[Callable[..., object]]:
    """The Activities an eval suite's scorers run in, or none for a non-suite class."""
    definitions: tuple[_ScorerDefinition, ...] = getattr(cls, _SUITE_ATTR, ())
    return [d.bind_activity() for d in definitions if d.kind == "activity"]


def is_eval_suite(cls: type) -> bool:
    return hasattr(cls, _SUITE_ATTR)


def _generated_suite_run(
    cls: type, definitions: tuple[_ScorerDefinition, ...]
) -> Callable[..., Awaitable[EvalSuiteResult]]:
    async def run(self: object, source: AgentTurn) -> EvalSuiteResult:
        capture = await workflow.execute_activity(
            CAPTURE_TRAJECTORY_ACTIVITY,
            source,
            result_type=CapturedTurn,
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=_RETRY,
        )
        outcomes = await asyncio.gather(
            *(_run_scorer(definition, definition.scope(capture)) for definition in definitions)
        )
        return EvalSuiteResult(
            source=source,
            results={d.name: outcome for d, outcome in zip(definitions, outcomes, strict=True)},
        )

    run.__qualname__ = f"{cls.__qualname__}.run"
    run.__module__ = cls.__module__
    run.__annotations__ = {"source": AgentTurn, "return": EvalSuiteResult}
    return workflow.run(run)


async def _run_scorer(
    definition: _ScorerDefinition, capture: CapturedTurn
) -> ScorerResult:
    try:
        if definition.kind == "inline":
            # Inline scorers share one capture, and its payloads are mutable even where the
            # event models are frozen, so each scores its own copy.
            return await definition.score(capture.model_copy(deep=True))
        return await workflow.execute_activity(
            definition.activity_type,
            capture,
            result_type=ScorerResult,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=_RETRY,
        )
    except Exception as exc:
        if is_cancelled_exception(exc):
            raise
        return _execution_error(exc.cause if isinstance(exc, ActivityError) and exc.cause else exc)


def _execution_error(exc: BaseException) -> ScorerResult:
    error_type = type(exc).__name__
    if isinstance(exc, ApplicationError):
        error_type = exc.type or error_type
        message = exc.message.strip()
    else:
        message = str(exc).strip()
    return ScorerResult(
        status="error",
        error_kind="execution",
        reasoning=f"{error_type}: {message}" if message else error_type,
    )
