"""Worker plugin that lets Workflows use the Claude Agent SDK and runs the Activities that reach Claude Code."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.metadata
import uuid
from collections.abc import Awaitable, Callable, Sequence
from types import MethodType
from typing import Any

from temporalio.plugin import SimplePlugin
from temporalio.worker import Worker, WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from temporal_agent_kit.ai_sdks.claude_agent_sdk._activities import TemporalTransportActivities
from temporal_agent_kit.ai_sdks.claude_agent_sdk._constants import DEFAULT_MAX_CONNECTIONS
from temporal_agent_kit.ai_sdks.claude_agent_sdk._subprocess import SubprocessConnector
from temporal_agent_kit.ai_sdks.claude_agent_sdk._types import Connector, EventPublisher, Output


class _NoopEventPublisher:
    """The event publisher for a plugin that isn't given one: publishes nothing."""

    def before_write(self) -> None:
        """Do nothing."""

    async def publish(self, message: Output) -> None:
        """Publish nothing."""


class ClaudeAgentSDKPlugin(SimplePlugin):
    """Use the Claude Agent SDK in Workflows.

    In the Workflow, pass the Claude Agent SDK a :class:`TemporalTransport` as
    its ``transport``. This plugin runs the transport's Activities, which reach
    Claude Code through ``connector``.

    Each application Worker Entity has its own private Worker Entity and task
    queue for its Claude Code connections. The plugin instance can be shared.
    """

    def __init__(
        self,
        *,
        connector: Connector | None = None,
        hook_activities: Sequence[Callable[..., Any]] = (),
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        event_publisher: type[EventPublisher] | None = None,
    ) -> None:
        """Create the plugin.

        Args:
            connector: Starts Claude Code and reaches it. Defaults to
                :class:`SubprocessConnector`, which runs it in a subprocess of
                the Worker Process, as the Claude Agent SDK does.
            hook_activities: Activities the transport's hooks run on the
                reserved private Task Queue, such as downloading files
                before Claude Code starts.
            max_connections: The most Claude Code connections each private
                Worker Entity runs at once. A Workflow that finds one full tries
                another one.
            event_publisher: The event publisher class. Each Claude Code connection
                gets its own instance, which its Activities call. By default, nothing
                is published.
        """
        if max_connections < 1:
            raise ValueError("max_connections must be at least 1.")

        self._hook_activities = hook_activities
        self._connector = connector if connector is not None else SubprocessConnector()
        self._max_connections = max_connections
        self._event_publisher: type[EventPublisher] = event_publisher or _NoopEventPublisher

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if runner is None:
                raise ValueError("No WorkflowRunner provided to ClaudeAgentSDKPlugin.")

            if isinstance(runner, SandboxedWorkflowRunner):
                # Import anyio's asyncio backend here, outside the sandbox.
                # Inside, anyio imports it lazily with import_module, which
                # skips the passthrough below and fails on its threading imports.
                importlib.import_module("anyio._backends._asyncio")

                return dataclasses.replace(
                    runner,
                    # Pass these through, so the Workflow shares one copy of
                    # them with the Worker. anyio, the Claude Agent SDK's async
                    # runtime, uses threads, and sniffio keeps the running
                    # async library in a thread-local.
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "claude_agent_sdk", "anyio", "sniffio"
                    ),
                )
            return runner

        super().__init__(
            name="ClaudeAgentSDKPlugin",
            workflow_runner=workflow_runner,
        )

    def version(self) -> str:
        """Return this package's version, which worker heartbeats report."""
        try:
            return importlib.metadata.version("temporal-agent-kit")
        except importlib.metadata.PackageNotFoundError:
            return ""

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Register the task queue reservation Activity on the application Worker Entity."""
        if _get_private_worker_entity_activities(config) is None:
            activities = TemporalTransportActivities(
                self._connector, f"claude-agent-sdk-{uuid.uuid4()}", self._max_connections, self._event_publisher
            )
            config["activities"] = [
                *config.get("activities", []),
                activities.reserve_private_task_queue,
            ]

        return super().configure_worker(config)

    async def run_worker(
        self, worker: Worker, next: Callable[[Worker], Awaitable[None]]
    ) -> None:
        """Run the private Worker Entity alongside the application Worker Entity, with its interceptors."""
        config = worker.config(active_config=True)
        activities = _get_private_worker_entity_activities(config)
        assert activities is not None

        if worker.task_queue == activities.private_task_queue:
            await next(worker)
            return

        transport_activities = [
            activities.connect,
            activities.write,
            activities.read,
            activities.end_input,
            activities.close,
        ]

        try:
            async with Worker(
                worker.client,
                task_queue=activities.private_task_queue,
                max_concurrent_activities=activities.max_connections * len(transport_activities),
                activities=[*transport_activities, *self._hook_activities],
                interceptors=config.get("interceptors", []),
            ):
                await super().run_worker(worker, next)
        finally:
            await asyncio.shield(activities.close_all())


def _get_private_worker_entity_activities(
    config: WorkerConfig,
) -> TemporalTransportActivities | None:
    for activity in config.get("activities", ()):
        if isinstance(activity, MethodType) and isinstance(activity.__self__, TemporalTransportActivities):
            return activity.__self__
    return None
