"""A ``ClaudeSDKClient`` that survives losing Claude Code."""

import dataclasses
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, Message, ResultMessage
from temporal_agent_kit.ai_sdks.claude_agent_sdk._transport import TemporalTransport
from temporalio.exceptions import ActivityError, ApplicationError

class TemporalClaudeSDKClient(ClaudeSDKClient):
    """A ``ClaudeSDKClient`` that reconnects and repeats the turn when Claude Code is lost."""

    def __init__(
        self,
        options: ClaudeAgentOptions | None = None,
        **transport_kwargs: Any,
    ) -> None:
        """Create the client. Claude Code starts on ``connect()``.

        Args:
            options: Claude Agent SDK options.
            **transport_kwargs: Passed to :class:`TemporalTransport`.
        """
        options = options if options is not None else ClaudeAgentOptions()

        self._session_id = options.resume
        self._reply_id = options.resume_session_at
        self._prompt: str | None = None

        self._temporal_transport = TemporalTransport(options, **transport_kwargs)
        super().__init__(options, transport=self._temporal_transport)

    async def connect(
        self, prompt: str | AsyncIterable[dict[str, Any]] | None = None
    ) -> None:
        """Start Claude Code after the last finished turn, retrying if it fails to start."""
        # Resume after the last finished turn, or start a new session if none finished.
        self.options = dataclasses.replace(self.options, resume=self._session_id, resume_session_at=self._reply_id)
        self._temporal_transport.options = self.options
        await super().connect(prompt)

    async def query(self, prompt: str | AsyncIterable[dict[str, Any]], session_id: str = "default") -> None:
        """Send a prompt, reconnecting and sending it again if Claude Code is lost."""
        if not isinstance(prompt, str):
            raise ApplicationError("Streamed prompts are not currently supported. Pass the prompt as a string.", non_retryable=True)

        self._prompt = prompt  # Sent again if Claude Code is lost before the result

        while True:
            try:
                await super().query(prompt, session_id)
                return
            except ActivityError:
                await self.disconnect()
                await self.connect()

    async def receive_messages(self) -> AsyncIterator[Message]:
        """Yield Claude Code's messages, reconnecting and repeating the turn if Claude Code is lost."""
        reply_id = None  # The latest top-level reply of the turn in progress
        while True:
            try:
                async for message in super().receive_messages():
                    if isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
                        reply_id = message.uuid
                    elif isinstance(message, ResultMessage):
                        self._session_id = message.session_id
                        self._reply_id = reply_id or self._reply_id
                        self._prompt = None
                        reply_id = None
                        await self._temporal_transport.after_turn()
                    yield message
                return
            except ActivityError:
                await self.disconnect()
                await self.connect(self._prompt)
                reply_id = None
