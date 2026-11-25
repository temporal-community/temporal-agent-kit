"""Authoring and dispatch for inline and Activity evals."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from temporalio import activity
from temporalio.exceptions import ApplicationError

from temporal_agent_kit.agent_events import AgentEvents
from temporal_agent_kit.agent_protocol import AgentEvent

__all__ = ["EvaluatorInput", "EvaluatorResult", "activity_eval_defn", "eval_defn"]


@dataclass(frozen=True)
class EvaluatorInput:
    """The requested turn ID and the retained events captured through that turn."""

    turn_id: str
    events: AgentEvents

    @property
    def turn_events(self) -> AgentEvents:
        """Events for the requested turn, including its subagents in merged order."""
        return self.events._for_turn_tree(self.turn_id)


class EvaluatorResult(BaseModel):
    """The outcome of one evaluator. A numeric value can have an inclusive range."""

    model_config = ConfigDict(allow_inf_nan=False)

    status: Literal["completed", "skipped", "error"]
    value: float | str | None = None
    value_range: tuple[float, float] | None = None
    reasoning: str | None = None
    error_kind: Literal["reported", "execution"] | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> EvaluatorResult:
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
            raise ValueError(
                "value_range lower bound must be less than or equal to its upper bound"
            )
        if not lower <= self.value <= upper:
            raise ValueError("value must be within value_range")
        return self


Evaluator = Callable[[EvaluatorInput], Awaitable[EvaluatorResult]]
_EvaluatorKind = Literal["inline", "activity"]
_EVALUATOR_ATTR = "__agent_evaluator__"


@dataclass(frozen=True)
class _EvaluatorDefinition:
    kind: _EvaluatorKind
    fn: Evaluator

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def activity_type(self) -> str:
        module = self.fn.__module__
        # Temporal loads an entry script under this name inside the Workflow sandbox.
        if module == "__temporal_main__":
            module = "__main__"
        return f"agentkit.eval.{module}.{self.fn.__qualname__}"

    def bind_activity(self) -> Callable[[list[AgentEvent], str], Awaitable[EvaluatorResult]]:
        @activity.defn(name=self.activity_type)
        async def run(events: list[AgentEvent], turn_id: str) -> EvaluatorResult:
            return await self.evaluate(EvaluatorInput(turn_id=turn_id, events=AgentEvents(events)))

        return run

    async def evaluate(self, evaluation: EvaluatorInput) -> EvaluatorResult:
        result = await self.fn(evaluation)
        try:
            # Validate a fresh snapshot because the author may have mutated a result model.
            outcome = EvaluatorResult.model_validate(
                result.model_dump() if isinstance(result, EvaluatorResult) else result
            )
        except ValidationError as exc:
            reason = exc.errors(include_input=False, include_url=False)[0]["msg"]
            raise ApplicationError(
                reason, type="InvalidEvaluatorResult", non_retryable=True
            ) from None
        if outcome.status == "error":
            outcome.error_kind = "reported"
        return outcome


@overload
def eval_defn(fn: Evaluator, /) -> Evaluator: ...


@overload
def eval_defn() -> Callable[[Evaluator], Evaluator]: ...


def eval_defn(fn: Evaluator | None = None, /) -> Evaluator | Callable[[Evaluator], Evaluator]:
    """Define an async evaluator that runs inline as deterministic eval suite Workflow code."""
    return _define_evaluator(fn, kind="inline")


@overload
def activity_eval_defn(fn: Evaluator, /) -> Evaluator: ...


@overload
def activity_eval_defn() -> Callable[[Evaluator], Evaluator]: ...


def activity_eval_defn(
    fn: Evaluator | None = None, /
) -> Evaluator | Callable[[Evaluator], Evaluator]:
    """Define an Activity evaluator over the retained history through the target turn."""
    return _define_evaluator(fn, kind="activity")


def _define_evaluator(
    fn: Evaluator | None, *, kind: _EvaluatorKind
) -> Evaluator | Callable[[Evaluator], Evaluator]:
    decorator = "@agent.eval_defn" if kind == "inline" else "@agent.activity_eval_defn"

    def decorate(evaluator: Evaluator) -> Evaluator:
        if hasattr(evaluator, _EVALUATOR_ATTR):
            raise TypeError(f"{decorator} requires a function with only one evaluator decorator")
        parameters = tuple(inspect.signature(evaluator).parameters.values())
        if (
            not inspect.iscoroutinefunction(evaluator)
            or len(parameters) != 1
            or parameters[0].kind
            not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ):
            raise TypeError(
                f"{decorator} requires an async function taking one EvaluatorInput"
            )
        hints = get_type_hints(evaluator)
        if (
            hints.get(parameters[0].name) is not EvaluatorInput
            or hints.get("return") is not EvaluatorResult
        ):
            raise TypeError(f"{decorator} requires (EvaluatorInput) -> EvaluatorResult")

        # A passed-through Activity only needs metadata that survives sandbox reimports.
        vars(evaluator)[_EVALUATOR_ATTR] = (
            kind if kind == "activity" else _EvaluatorDefinition(kind, evaluator)
        )
        return evaluator

    return decorate(fn) if fn is not None else decorate
