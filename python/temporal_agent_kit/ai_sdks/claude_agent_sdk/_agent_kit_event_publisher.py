"""Publishes Claude Code's messages as Agent Kit events from the transport's Activities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import StreamEvent
from claude_agent_sdk._internal.message_parser import parse_message

from temporal_agent_kit.agent_protocol import (
    AgentStreamItem,
    ModelInteractionEnded,
    ModelInteractionStarted,
    ReplyDelta,
    TextAnnotationDelta,
    ThoughtSummaryDelta,
    TokenUsage,
)
from temporal_agent_kit.agent_workflow import _CURRENT_STREAM_CONTEXT, AgentWorkflowRunner
from temporal_agent_kit.ai_sdks.claude_agent_sdk._types import Output
from temporal_agent_kit.stream_context import TurnStreamContext


@dataclass
class _ModelCall:
    """The model call in progress."""

    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def add_usage(self, usage: dict[str, Any] | None) -> None:
        """Record the token counts in a usage. Claude sends running totals, so each replaces the last."""
        if usage is None:
            return
        assert isinstance(usage, dict)

        # Usage also holds non-count fields, such as server_tool_use and cache_creation.
        for key, value in usage.items():
            if isinstance(value, int):
                self.usage[key] = value

    def ended(self) -> ModelInteractionEnded:
        """The event that closes this call."""
        return ModelInteractionEnded(model=self.model, usage=self._token_usage())

    def _token_usage(self) -> TokenUsage | None:
        """Convert Claude's usage to TokenUsage, adding cached tokens back into the input."""
        if not self.usage:
            return None

        input_tokens = (
            self.usage.get("input_tokens", 0)
            + self.usage.get("cache_read_input_tokens", 0)
            + self.usage.get("cache_creation_input_tokens", 0)
        )
        output_tokens = self.usage.get("output_tokens")
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=self.usage.get("cache_read_input_tokens"),
            total_tokens=input_tokens + (output_tokens or 0),
        )


class AgentKitEventPublisher:
    """Publishes Agent Kit events for Claude Code."""

    def __init__(self) -> None:
        self._context: TurnStreamContext | None = None
        self._model_call = _ModelCall()

    def before_write(self) -> None:
        """Store the current context for publish to use."""
        context = _CURRENT_STREAM_CONTEXT.get()
        if context is not None:
            self._context = context

    async def publish(self, message: Output) -> None:
        """Publish the Agent Kit events for one message from Claude Code."""
        match message.get("type"):
            case "stream_event":
                await self._publish_stream_event_message(message)

    async def _publish(self, event: AgentStreamItem) -> None:
        """Add an event for the message being published."""
        if self._context:
            async with AgentWorkflowRunner.publisher_from_activity(self._context) as publisher:
                publisher.publish(event)

    async def _publish_stream_event_message(self, message: Output) -> None:
        """Publish the events for a stream_event message, which wraps one Anthropic API stream event."""
        stream_event = parse_message(message)
        assert isinstance(stream_event, StreamEvent)
        event = stream_event.event

        match event["type"]:
            case "message_start":
                self._model_call = _ModelCall(model=event["message"].get("model"))
                self._model_call.add_usage(event["message"].get("usage"))
                await self._publish(ModelInteractionStarted())
            case "message_delta":
                self._model_call.add_usage(event.get("usage"))
            case "message_stop":
                await self._publish(self._model_call.ended())
                self._model_call = _ModelCall()
            case "content_block_delta":
                await self._publish_content_block_delta(event["delta"])

    async def _publish_content_block_delta(self, delta: dict[str, Any]) -> None:
        """Publish the event for one streamed piece of a content block."""
        match delta["type"]:
            case "text_delta":
                await self._publish(ReplyDelta(text=delta["text"]))
            case "thinking_delta":
                # Thinking text uses the thought summary's text field.
                await self._publish(ThoughtSummaryDelta(delta={"type": "thinking_delta", "text": delta["thinking"]}))
            case "signature_delta":
                # The signature is opaque metadata for the thinking block.
                await self._publish(ThoughtSummaryDelta(delta={"type": "signature_delta", "signature": delta["signature"]}))
            case "citations_delta":
                # Attach the citation to the streamed reply as an annotation.
                await self._publish(TextAnnotationDelta(delta={"annotations": [delta["citation"]]}))
