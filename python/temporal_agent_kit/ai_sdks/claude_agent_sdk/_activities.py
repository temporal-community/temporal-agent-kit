"""The Activities a :class:`TemporalTransport` runs, which reach Claude Code through a :class:`Connector`."""

from __future__ import annotations

import asyncio
import logging

from claude_agent_sdk import Transport
from temporalio import activity
from temporalio.exceptions import ApplicationError

from temporal_agent_kit.ai_sdks.claude_agent_sdk._constants import (
    CLOSE,
    CONNECT,
    END_INPUT,
    READ,
    RESERVE_PRIVATE_TASK_QUEUE,
    WRITE,
)
from temporal_agent_kit.ai_sdks.claude_agent_sdk._heartbeat_decorator import (
    auto_heartbeater,
)
from temporal_agent_kit.ai_sdks.claude_agent_sdk._types import (
    ConnectActivityInput,
    Connector,
    ReadActivityInput,
    EventPublisher,
    WriteActivityInput,
)

logger = logging.getLogger(__name__)


class TemporalTransportActivities:
    """Route transport Activities to the private Worker Entity's connections."""

    def __init__(
        self,
        connector: Connector,
        private_task_queue: str,
        max_connections: int,
        event_publisher: type[EventPublisher],
    ) -> None:
        self.private_task_queue = private_task_queue
        self.max_connections = max_connections
        self._connector = connector
        self._reserve_lock = asyncio.Lock()
        self._connections: dict[str, _ClaudeCodeConnection] = {}
        self._event_publisher = event_publisher

    @activity.defn(name=RESERVE_PRIVATE_TASK_QUEUE)
    @auto_heartbeater
    async def reserve_private_task_queue(self, connection_id: str) -> str:
        """Reserve a connection slot and return its private task queue."""
        async with self._reserve_lock:
            if connection_id in self._connections:
                return self.private_task_queue

            if len(self._connections) >= self.max_connections:
                raise ApplicationError(
                    f"This private Worker Entity already runs its limit of {self.max_connections} Claude Code connections.",
                    non_retryable=True,
                )

            self._connections[connection_id] = _ClaudeCodeConnection(self._connector, self._event_publisher())
            return self.private_task_queue

    @activity.defn(name=CONNECT)
    @auto_heartbeater
    async def connect(self, connection_id: str, request: ConnectActivityInput) -> None:
        await self._get_connection(connection_id).connect(request)

    @activity.defn(name=WRITE)
    @auto_heartbeater
    async def write(self, connection_id: str, request: WriteActivityInput) -> None:
        await self._get_connection(connection_id).write(request)

    @activity.defn(name=END_INPUT)
    @auto_heartbeater
    async def end_input(self, connection_id: str) -> None:
        await self._get_connection(connection_id).end_input()

    @activity.defn(name=READ)
    @auto_heartbeater
    async def read(self, connection_id: str, request: ReadActivityInput) -> None:
        await self._get_connection(connection_id).read(request)

    @activity.defn(name=CLOSE)
    @auto_heartbeater
    async def close(self, connection_id: str) -> None:
        connection = self._connections.get(connection_id)
        if connection is None:
            return

        await connection.close()
        self._connections.pop(connection_id, None)

    async def close_all(self) -> None:
        results = await asyncio.gather(
            *(self.close(connection_id) for connection_id in self._connections),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                # Log rather than raise, so the Worker's own error isn't hidden.
                logger.error("Failed to close a Claude Code connection.", exc_info=result)

    def _get_connection(self, connection_id: str) -> _ClaudeCodeConnection:
        connection = self._connections.get(connection_id)
        if connection is None:
            raise ApplicationError("Claude Code connection is not reserved.", non_retryable=True)
        return connection


class _ClaudeCodeConnection:
    """The process for one reserved connection."""

    def __init__(self, connector: Connector, event_publisher: EventPublisher) -> None:
        self._connector = connector
        self._event_publisher = event_publisher
        self._close_lock = asyncio.Lock()
        self._transport: Transport | None = None

    async def connect(self, request: ConnectActivityInput) -> None:
        """Start Claude Code for the reservation."""
        if self._transport is None:
            self._transport = await self._connector.connect(request)

    async def write(self, request: WriteActivityInput) -> None:
        """Write to Claude Code's stdin."""
        transport = self._running_transport()
        self._event_publisher.before_write()
        await transport.write(request.text)

    async def end_input(self) -> None:
        """Close Claude Code's stdin."""
        await self._running_transport().end_input()

    async def read(self, request: ReadActivityInput) -> None:
        """Signal each message from Claude Code to the Workflow, and fail when its output ends."""
        info = activity.info()
        if info.workflow_id is None:
            raise ApplicationError("The read Activity must be started by a Workflow.", non_retryable=True)

        handle = activity.client().get_workflow_handle(info.workflow_id, run_id=info.workflow_run_id)

        transport = self._running_transport()

        async for message in transport.read_messages():
            await self._event_publisher.publish(message)
            await handle.signal(request.signal_name, [message])

        raise ApplicationError("Claude Code's output ended.")

    async def close(self) -> None:
        """Close the process if it has started."""
        async with self._close_lock:
            if self._transport is not None:
                await self._transport.close()
                self._transport = None

    def _running_transport(self) -> Transport:
        if self._transport is None:
            raise ApplicationError("No Claude Code connection established.", non_retryable=True)
        return self._transport
