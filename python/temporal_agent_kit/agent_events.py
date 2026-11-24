"""Read-only views over the native events supplied to Temporal Agent Kit evals."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeVar

import jsonpatch

from temporal_agent_kit.agent_protocol.agent_interface import CloseReason
from temporal_agent_kit.agent_protocol.events import (
    AgentClosed,
    AgentEvent,
    AgentStatePatch,
    AgentStateSnapshot,
    AutoApprovalEvaluationError,
    CallbackRequested,
    CallbackResolved,
    MessageHandlerEnd,
    MessageHandlerError,
    ModelInteractionEnded,
    ModelInteractionStarted,
    SteeringEnqueued,
    StreamEvent,
    SubagentMessageSent,
    SubagentReplyReceived,
    SubagentStarted,
    SubagentStopped,
    ThoughtSummaryDelta,
    TokenUsage,
    ToolApprovalRequested,
    ToolApprovalResolved,
    ToolEndEvent,
    ToolErrorEvent,
    ToolEvent,
    ToolRequested,
    ToolStartEvent,
)

__all__ = ["AgentEvents", "ToolCall"]

_Event = TypeVar("_Event", bound=StreamEvent[Any])


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The observed lifecycle of one tool invocation, in trajectory order."""

    agent_id: str
    turn_id: str
    message_id: str | None
    tool_id: str
    tool_name: str
    events: tuple[AgentEvent, ...]

    def _last(self, event_type: type[_Event]) -> _Event | None:
        return next(
            (item.event for item in reversed(self.events) if isinstance(item.event, event_type)),
            None,
        )

    @property
    def requested(self) -> ToolRequested | None:
        return self._last(ToolRequested)

    @property
    def approval_requested(self) -> ToolApprovalRequested | None:
        return self._last(ToolApprovalRequested)

    @property
    def approval_resolved(self) -> ToolApprovalResolved | None:
        return self._last(ToolApprovalResolved)

    @property
    def started(self) -> ToolStartEvent | None:
        return self._last(ToolStartEvent)

    @property
    def ended(self) -> ToolEndEvent | None:
        return self._last(ToolEndEvent)

    @property
    def error(self) -> ToolErrorEvent | None:
        return self._last(ToolErrorEvent)

    @property
    def callback_requested(self) -> CallbackRequested | None:
        return self._last(CallbackRequested)

    @property
    def callback_resolved(self) -> CallbackResolved | None:
        return self._last(CallbackResolved)


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

    def root_only(self) -> AgentEvents:
        """Keep events emitted by the root, identified by its single-segment agent ID."""
        return self._view(item for item in self._events if "-" not in item.agent_id)

    def for_agent(self, agent_id: str) -> AgentEvents:
        """Keep events emitted by this exact agent ID."""
        return self._view(item for item in self._events if item.agent_id == agent_id)

    def for_turn(self, turn_id: str) -> AgentEvents:
        """Keep the exact turn ID on the envelope. Child turns have their own IDs."""
        return self._view(item for item in self._events if item.turn_id == turn_id)

    def for_message(self, message_id: str) -> AgentEvents:
        """Keep events attributed to this exact message ID."""
        return self._view(item for item in self._events if item.message_id == message_id)

    def subagents(self, agent_key: str | None = None) -> list[AgentEvents]:
        """One view per observed descendant, ordered by its first event.

        Filtering by key requires a parent event in this view naming the descendant's key.
        """
        keys: dict[str, str] = {}
        if agent_key is not None:
            for item in self._events:
                event = item.event
                if isinstance(
                    event,
                    (SubagentStarted, SubagentStopped, SubagentMessageSent, SubagentReplyReceived),
                ):
                    keys[event.subagent_id] = event.agent_key

        groups: dict[str, list[AgentEvent]] = {}
        for item in self._events:
            if "-" in item.agent_id and (agent_key is None or keys.get(item.agent_id) == agent_key):
                groups.setdefault(item.agent_id, []).append(item)
        return [self._view(items) for items in groups.values()]

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
    def thought_summaries(self) -> list[dict[str, Any]]:
        """Observed model thought-summary chunks, in order."""
        return [event.delta for event in self._payloads_of_type(ThoughtSummaryDelta)]

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Tool calls, including partial calls, ordered by their first observed event."""
        groups: dict[tuple[str, str, str | None, str], tuple[str, list[AgentEvent]]] = {}
        for item in self._events:
            event = item.event
            if isinstance(event, ToolEvent):
                key = item.agent_id, item.turn_id, item.message_id, event.tool_id
                _, items = groups.setdefault(key, (event.tool_name, []))
                items.append(item)
        return [ToolCall(*key, name, tuple(items)) for key, (name, items) in groups.items()]

    def tool_calls_named(self, name: str) -> list[ToolCall]:
        return [call for call in self.tool_calls if call.tool_name == name]

    @property
    def rejected_tool_calls(self) -> list[ToolCall]:
        """Calls with an observed denial at the approval gate."""
        return [
            call
            for call in self.tool_calls
            if (decision := call.approval_resolved) is not None and not decision.approved
        ]

    @property
    def callbacks(self) -> list[ToolCall]:
        """Calls with an observed callback request or resolution."""
        return [
            call
            for call in self.tool_calls
            if call.callback_requested is not None or call.callback_resolved is not None
        ]

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
    def steering_messages(self) -> list[str]:
        """Enqueued steering text, including messages later dropped or still pending."""
        return [event.text for event in self._payloads_of_type(SteeringEnqueued)]

    def final_state(self, state_id: str) -> dict[str, Any] | None:
        """Reconstruct the state from its latest snapshot and consecutive patches.

        Returns ``None`` for an absent snapshot or a version gap. Raises ``ValueError``
        if the state ID appears on multiple agents in the view.
        """
        agent_id: str | None = None
        snapshot: AgentStateSnapshot | None = None
        patches: list[AgentStatePatch] = []
        for item in self._events:
            event = item.event
            if (
                not isinstance(event, (AgentStateSnapshot, AgentStatePatch))
                or event.state_id != state_id
            ):
                continue
            if agent_id is not None and agent_id != item.agent_id:
                raise ValueError(f"state {state_id!r} appears on multiple agents; select an agent")
            agent_id = item.agent_id
            if isinstance(event, AgentStateSnapshot):
                snapshot = event
                patches.clear()
            else:
                patches.append(event)

        if snapshot is None:
            return None
        ops: list[dict[str, Any]] = []
        for version, patch in enumerate(patches, start=snapshot.version + 1):
            if patch.version != version:
                return None
            ops.extend(patch.ops)
        # Patch ops can insert mutable values that later ops modify in place.
        return jsonpatch.apply_patch(snapshot.value, deepcopy(ops))

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
