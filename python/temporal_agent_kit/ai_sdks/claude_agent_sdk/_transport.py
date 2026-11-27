"""A Claude Agent SDK ``Transport`` for Workflows, which reaches Claude Code through Activities."""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import timedelta
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, Transport
from claude_agent_sdk import __version__ as sdk_version
from claude_agent_sdk._internal.transport.subprocess_cli import (
    _SDK_READS_SESSION_STATE_ENV,
    SubprocessCLITransport,
)
from claude_agent_sdk.types import _configure_can_use_tool
from temporalio import workflow
from temporalio.exceptions import ActivityError, ApplicationError, CancelledError

from temporal_agent_kit.ai_sdks.claude_agent_sdk._constants import (
    CLOSE,
    CLOSE_OPTIONS,
    CONNECT,
    CONNECT_OPTIONS,
    END_INPUT,
    END_INPUT_OPTIONS,
    MAX_WORKER_CONNECT_ATTEMPTS,
    READ,
    READ_OPTIONS,
    RESERVE_PRIVATE_TASK_QUEUE,
    RESERVE_PRIVATE_TASK_QUEUE_OPTIONS,
    STDIN_LABELS,
    WRITE,
    WRITE_OPTIONS,
)
from temporal_agent_kit.ai_sdks.claude_agent_sdk._types import (
    ConnectActivityInput,
    Output,
    ReadActivityInput,
    TransportActivityOptions,
    TransportHooks,
    WriteActivityInput,
)


class TemporalTransport(Transport):
    """A Claude Agent SDK ``Transport`` for Workflows that uses Activities for process I/O."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        *,
        hooks: TransportHooks | None = None,
        activity_options: TransportActivityOptions | None = None,
    ) -> None:
        """Prepare the transport. Claude Code starts on ``connect``.

        Args:
            options: The options passed to ``ClaudeSDKClient``.
            hooks: Workflow callbacks run at points in Claude Code's lifecycle.
            activity_options: Options for each Activity.
        """
        if options.stderr is not None:
            raise ValueError(
                "Configure stderr through the plugin's connector. "
                "Leave ClaudeAgentOptions.stderr unset in the Workflow."
            )

        if not self._is_json_object(options.settings):
            raise ValueError(
                f"{type(self).__name__} needs options.settings to be a JSON "
                "object, not a file path"
            )

        self.options = options
        """The options Claude Code starts with. Change them before ``connect()`` to start it differently, such as resuming a session."""
        self._activity_options: TransportActivityOptions = activity_options or {}

        self._task_queue: str | None = None
        self._connection_id: str | None = None
        self._connected = False

        self._write_lock = asyncio.Lock()

        self._output: list[Output] = []

        self._hooks: TransportHooks = hooks or {}

        # Map replay-safe request IDs back to the Claude Agent SDK's original
        # request IDs when the CLI answers.
        self._sdk_ids: dict[str, str] = {}

    async def connect(self) -> None:
        """Get a private task queue, run setup, then start Claude Code."""
        self._output = []
        self._sdk_ids = {}

        for attempt in range(MAX_WORKER_CONNECT_ATTEMPTS):
            try:
                self._connection_id = self._workflow_uuid()

                queue_options: workflow.ActivityConfig = {
                    **RESERVE_PRIVATE_TASK_QUEUE_OPTIONS,
                    **self._activity_options.get("reserve", {}),
                }
                self._task_queue = await workflow.execute_activity(
                    RESERVE_PRIVATE_TASK_QUEUE, self._connection_id, result_type=str, **queue_options
                )
                assert self._task_queue is not None

                if (before_connect := self._hooks.get("before_connect")) is not None:
                    await before_connect(self._task_queue)

                options: workflow.ActivityConfig = {
                    **CONNECT_OPTIONS,
                    **self._activity_options.get("connect", {}),
                    "task_queue": self._task_queue,
                }
                await workflow.execute_activity(
                    CONNECT, args=[self._connection_id, self._connect_input()], **options
                )

                self._connected = True
                return
            except ActivityError as error:
                await self.close()

                if attempt == MAX_WORKER_CONNECT_ATTEMPTS - 1:
                    raise ApplicationError(
                        f"Could not connect to Claude Code after {MAX_WORKER_CONNECT_ATTEMPTS} attempts.",
                        non_retryable=True,
                    ) from error

                # Try a different private Worker Entity
                await workflow.sleep(timedelta(seconds=30))

    def is_ready(self) -> bool:
        """Return whether the transport has an open connection."""
        return self._connected

    async def write(self, data: str) -> None:
        """Send the Claude Agent SDK's prompts, requests, and answers to the CLI."""
        if not self._connected or not self._task_queue:
            raise ApplicationError("A connection to a Claude Code process must be established before writing.")

        async with self._write_lock:
            message = json.loads(data)
            kind: str = message["type"]

            # The request id needs to be swapped for a deterministic one for Replay
            if kind == "control_request":
                request_id = self._workflow_uuid()
                self._sdk_ids[request_id] = message["request_id"]
                data = f"{json.dumps({**message, 'request_id': request_id})}\n"
            elif kind == "user" and "uuid" not in message:
                # Claude Code skips a prompt whose uuid it already has, so a
                # retried write doesn't run it again.
                data = f"{json.dumps({**message, 'uuid': self._workflow_uuid()})}\n"

            if (before_write := self._hooks.get("before_write")) is not None:
                await before_write(self._task_queue)

            options: workflow.ActivityConfig = {
                **WRITE_OPTIONS,
                "summary": f"Send {STDIN_LABELS.get(kind, kind)}",
                **self._activity_options.get("write", {}),
                "task_queue": self._task_queue,
            }
            try:
                await workflow.execute_activity(
                    WRITE,
                    args=[self._connection_id, WriteActivityInput(text=data)],
                    **options,
                )
            except ActivityError:
                if kind != "control_response":
                    raise
                # The Claude Agent SDK doesn't handle errors raised while writing a control
                # response, so catch it here to avoid an uncaught error.
                workflow.logger.warning("Dropped an answer to Claude Code: its connection is unreachable.")

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield messages from Claude Code as they arrive."""
        if not self._connected or not self._task_queue:
            raise ApplicationError("A connection to a Claude Code process must be established before reading.")

        signal_name = f"{READ}-{self._connection_id}"
        workflow.set_signal_handler(signal_name, self._receive_output)

        read_options: workflow.ActivityConfig = {
            **READ_OPTIONS,
            **self._activity_options.get("read", {}),
            "task_queue": self._task_queue,
        }
        read = workflow.start_activity(
            READ, args=[self._connection_id, ReadActivityInput(signal_name=signal_name)], **read_options
        )

        while True:
            try:
                await workflow.wait_condition(lambda: bool(self._output) or read.done())
            except asyncio.CancelledError:
                read.cancel()
                raise

            try:
                if not self._output:
                    await read
            except ActivityError as e:
                if isinstance(e.cause, CancelledError):
                    return
                raise

            outputs, self._output = self._output, []
            for message in self._handle_output(outputs):
                if (on_output := self._hooks.get("on_output")) is not None:
                    await on_output(self._task_queue)
                yield message

    async def end_input(self) -> None:
        """End the CLI's stdin, after the writes before it."""
        if not self._connected or not self._task_queue:
            return

        async with self._write_lock:
            options: workflow.ActivityConfig = {
                **END_INPUT_OPTIONS,
                **self._activity_options.get("end_input", {}),
                "task_queue": self._task_queue,
            }

            await workflow.execute_activity(END_INPUT, self._connection_id, **options)

    async def close(self) -> None:
        """Stop the CLI and everything it started.

        If the close Activity fails, for example because Claude Code's Worker is gone, log it and
        forget the connection anyway. It also releases the slot a failed ``connect()`` attempt reserved,
        which is why it does not require the connection to be established.
        """
        if not self._task_queue or not self._connection_id:
            return

        options: workflow.ActivityConfig = {
            **CLOSE_OPTIONS,
            **self._activity_options.get("close", {}),
            "task_queue": self._task_queue,
        }

        try:
            await workflow.execute_activity(CLOSE, self._connection_id, **options)

        except ActivityError:
            workflow.logger.warning("Failed to disconnect from Claude Code.", exc_info=True)

        self._task_queue = None
        self._connection_id = None
        self._connected = False

    async def after_turn(self) -> None:
        if (after_turn := self._hooks.get("after_turn")) is not None and self._task_queue:
            await after_turn(self._task_queue)

    def _receive_output(self, batch: list[Output]) -> None:
        """Take the messages one Signal from the read Activity carries."""
        self._output.extend(batch)

    def _handle_output(self, outputs: list[Output]) -> Iterator[dict[str, Any]]:
        """Handle a batch of the CLI's output, and yield the messages in it."""
        for output in outputs:
            if output.get("type") != "control_response":
                yield output
                continue

            # Swap the deterministic request ID back to the Claude Agent SDK's.
            # Pass other responses as they are, such as the SDK's own answers that
            # replay-user-messages echoes, which the SDK ignores.
            response = output["response"]
            sdk_id = self._sdk_ids.pop(response["request_id"], None)
            if sdk_id is None:
                yield output
                continue

            yield {**output, "response": {**response, "request_id": sdk_id}}

    def _connect_input(self) -> ConnectActivityInput:
        """Claude Code's command line and environment, built from the options as in the Claude Agent SDK."""
        options = _configure_can_use_tool(self.options)
        # The Claude Agent SDK builds the command line only inside its
        # subprocess transport, so that code is borrowed here. Replace this once
        # the Python Claude Agent SDK has a hook like the TypeScript one's
        # spawnClaudeCodeProcess. It needs a CLI path, which only the connector
        # can find, so it gets a placeholder that is then dropped.
        placeholder = dataclasses.replace(options, cli_path="claude")
        args = SubprocessCLITransport(prompt="", options=placeholder)._build_command()[
            1:
        ]

        env = {
            "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
            **options.env,
            "CLAUDE_AGENT_SDK_VERSION": sdk_version,
        }

        # As in the Claude Agent SDK, set it unless options.env has it, in any
        # letter case. It makes Claude Code emit the session_state_changed
        # messages the Claude Agent SDK uses to tell when a run is over.
        if not any(key.upper() == _SDK_READS_SESSION_STATE_ENV for key in env):
            env[_SDK_READS_SESSION_STATE_ENV] = "1"

        if options.enable_file_checkpointing:
            env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"

        return ConnectActivityInput(
            args,
            env,
            cli_path=str(options.cli_path) if options.cli_path else None,
            cwd=str(options.cwd) if options.cwd else None,
            user=options.user,
            max_buffer_size=options.max_buffer_size,
        )

    def _workflow_uuid(self) -> str:
        instance = workflow.instance()
        random = getattr(instance, "__claude_agent_sdk_random", None)
        if random is None:
            random = workflow.new_random("claude-agent-sdk-random")
            setattr(instance, "__claude_agent_sdk_random", random)
        return str(uuid.UUID(int=random.getrandbits(128), version=4))

    def _is_json_object(self, settings: str | None) -> bool:
        """Whether ``settings`` is unset or a JSON object, which the Claude Agent SDK merges without reading a file."""
        if settings is None:
            return True
        try:
            return isinstance(json.loads(settings), dict)
        except ValueError:
            return False
