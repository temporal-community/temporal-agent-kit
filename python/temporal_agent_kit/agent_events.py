"""Read-only views over the native events supplied to Temporal Agent Kit evals."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from temporal_agent_kit.agent_protocol.agent_interface import CloseReason
from temporal_agent_kit.agent_protocol.events import (
    AgentClosed,
    AgentEvent,
    AutoApprovalEvaluationError,
    MessageHandlerEnd,
    MessageHandlerError,
    ModelInteractionEnded,
    ModelInteractionStarted,
    StreamEvent,
    TokenUsage,
    ToolErrorEvent,
)

__all__ = ["AgentEvents"]

_Event = TypeVar("_Event", bound=StreamEvent[Any])


class AgentEvents:
    """An ordered view of supplied native events from one agent tree.

    The iterable is consumed once. Views share the original envelopes and payloads,
    which callers should treat as read-only.
    """

    def __init__(self, events: Iterable[AgentEvent]) -> None:
        self._events = tuple(events)

    def _view(self, events: Iterable[AgentEvent]) -> AgentEvents:
        return AgentEvents(events)

    @property
    def all(self) -> list[AgentEvent]:
        """Native envelopes in their supplied order."""
        return list(self._events)

    def of_type(self, *types: type[StreamEvent[Any]]) -> AgentEvents:
        """Keep envelopes whose payload matches any of the given classes, in order."""
        return self._view(item for item in self._events if isinstance(item.event, types))

    def _payloads_of_type(self, event_type: type[_Event]) -> list[_Event]:
        return [item.event for item in self._events if isinstance(item.event, event_type)]

    def for_agent(self, agent_id: str) -> AgentEvents:
        """Keep events emitted by this exact agent ID."""
        return self._view(item for item in self._events if item.agent_id == agent_id)

    def for_turn(self, turn_id: str) -> AgentEvents:
        """Keep the exact turn ID on the envelope. Child turns have their own IDs."""
        return self._view(item for item in self._events if item.turn_id == turn_id)

    def for_message(self, message_id: str) -> AgentEvents:
        """Keep events attributed to this exact message ID."""
        return self._view(item for item in self._events if item.message_id == message_id)

    @property
    def replies(self) -> list[AgentEvent]:
        """Successful handler outputs with their agent, turn, and message envelopes."""
        return self.of_type(MessageHandlerEnd).all

    @property
    def reply_text(self) -> str | None:
        """The sole completed reply's string ``text`` field, when present.

        Raises ``ValueError`` for multiple completed replies. Streamed deltas are available
        separately via ``of_type(ReplyDelta)``.
        """
        replies = self._payloads_of_type(MessageHandlerEnd)
        if len(replies) > 1:
            raise ValueError("multiple completed replies; select an agent, turn, or message")
        if not replies:
            return None
        value = replies[0].output.get("text")
        return value if isinstance(value, str) else None

    @property
    def model_call_count(self) -> int:
        """Number of observed model interaction starts, including unfinished calls."""
        return len(self._payloads_of_type(ModelInteractionStarted))

    @property
    def total_usage(self) -> TokenUsage:
        """Sum each field reported by every observed completion.

        Fields are unknown when any completion omits them. With no completions or with
        an observed start still pending, every field is unknown.
        """
        pending: dict[tuple[str, str, str | None], int] = {}
        completed: list[TokenUsage] = []
        for item in self._events:
            event = item.event
            key = item.agent_id, item.turn_id, item.message_id
            if isinstance(event, ModelInteractionStarted):
                pending[key] = pending.get(key, 0) + 1
            elif isinstance(event, ModelInteractionEnded):
                if pending.get(key, 0):
                    pending[key] -= 1
                if event.usage is None:
                    return TokenUsage()
                completed.append(event.usage)

        totals: dict[str, int] = {}
        if completed and not any(pending.values()):
            for field in TokenUsage.model_fields:
                counts = [getattr(usage, field) for usage in completed]
                if all(count is not None for count in counts):
                    totals[field] = sum(counts)
        return TokenUsage(**totals)

    @property
    def errors(self) -> list[str]:
        """Handler, tool, and automatic approval evaluation error messages in order."""
        return [
            event.message
            for item in self._events
            if isinstance(
                event := item.event, (MessageHandlerError, ToolErrorEvent, AutoApprovalEvaluationError)
            )
        ]

    @property
    def close_reason(self) -> CloseReason | None:
        """The sole observed close reason. Raises ``ValueError`` for multiple closes."""
        closes = self._payloads_of_type(AgentClosed)
        if len(closes) > 1:
            raise ValueError("multiple agent closes; select an agent")
        return closes[0].reason if closes else None
