"""A ``ClaudeSDKClient`` wired to Agent Kit streaming and interrupt recovery."""

from __future__ import annotations

import dataclasses
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions
from temporal_agent_kit.stream_context import TurnStreamContext

from temporal_agent_kit.agent_protocol import (
    AgentEventType,
    AgentStreamItem,
    ModelInteractionEnded,
)
from temporal_agent_kit.agent_workflow import (
    _CURRENT_STREAM_CONTEXT,
    _CURRENT_MESSAGE_ID,
    AgentWorkflowRunner,
)

from temporal_agent_kit.ai_sdks.claude_agent_sdk._client import TemporalClaudeSDKClient
from temporal_agent_kit.ai_sdks.claude_agent_sdk._types import TransportHooks



class AgentKitClaudeSDKClient(TemporalClaudeSDKClient):
    """A :class:`TemporalClaudeSDKClient` wired to stream turns and handle interrupts with Agent Kit."""

    def __init__(
        self,
        options: ClaudeAgentOptions | None = None,
        *,
        runner: AgentWorkflowRunner,
        **transport_kwargs: Any,
    ) -> None:
        """Prepare the client. Claude Code starts on ``connect()``.

        Args:
            options: Claude Agent SDK options. Defaults to ``ClaudeAgentOptions()``.
            runner: The Agent Kit runner to stream turns to.
            **transport_kwargs: ``hooks`` and ``activity_options`` for the :class:`TemporalTransport`.
        """
        options = options or ClaudeAgentOptions()
        self._runner = runner

        self._context: TurnStreamContext | None = None

        super().__init__(
            self._options(options),
            hooks=self._transport_hooks(transport_kwargs.pop("hooks", None)),
            **transport_kwargs,
        )

    async def disconnect(self) -> None:
        """End the model call and close the connection."""
        if self._query is not None and self._runner.current_stream_context is not None:
            self._publish(ModelInteractionEnded())
        await super().disconnect()

    def _options(self, base: ClaudeAgentOptions) -> ClaudeAgentOptions:
        overrides: dict[str, Any] = {"include_partial_messages": True}
        if base.thinking is None and base.max_thinking_tokens is None:
            overrides["thinking"] = {"type": "adaptive", "display": "summarized"}
        return dataclasses.replace(base, **overrides)

    def _transport_hooks(self, base: TransportHooks | None) -> TransportHooks:
        hooks: TransportHooks = {**(base or {})}

        original_before_write = hooks.get("before_write")
        async def before_write(task_queue: str) -> None:
            context = _CURRENT_STREAM_CONTEXT.get()
            if context is not None:
                self._context = context

            if original_before_write is not None:
                await original_before_write(task_queue)
        hooks["before_write"] = before_write

        original_on_output = hooks.get("on_output")
        async def on_output(task_queue: str) -> None:
            if self._context is not None:
                _CURRENT_STREAM_CONTEXT.set(self._context)
                _CURRENT_MESSAGE_ID.set(self._context.message_id)

            if original_on_output is not None:
                await original_on_output(task_queue)
        hooks["on_output"] = on_output

        return hooks

    def _publish(self, event: AgentStreamItem):
        if self._context:
            self._runner._pub(
                self._context.turn_id, self._context.turn_number, event, message_id=self._context.message_id
            )
