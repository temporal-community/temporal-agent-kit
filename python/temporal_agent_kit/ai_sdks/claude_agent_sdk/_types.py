"""The types shared by a :class:`TemporalTransport`, its Activities, and how a Worker Process reaches Claude Code."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from claude_agent_sdk import Transport
from temporalio import workflow

Output = dict[str, Any]
"""A message from Claude Code's stdout."""


class TransportActivityOptions(TypedDict, total=False):
    """Options for each Activity a :class:`TemporalTransport` runs, overriding its own.

    Set ``task_queue`` in ``reserve`` to choose where the reservation Activity
    runs. All subsequent Activities run on the reserved private Task Queue.
    """

    reserve: workflow.ActivityConfig
    """Reserves a Claude Code connection and its private Task Queue."""
    connect: workflow.ActivityConfig
    """Starts Claude Code."""
    write: workflow.ActivityConfig
    """Writes to Claude Code's stdin."""
    end_input: workflow.ActivityConfig
    """Closes Claude Code's stdin."""
    read: workflow.ActivityConfig
    """Signals each message from Claude Code to the Workflow until its output ends or the connection closes."""
    close: workflow.ActivityConfig
    """Closes Claude Code."""


class TransportHooks(TypedDict, total=False):
    """Workflow callbacks run at points in Claude Code's lifecycle."""

    before_connect: Callable[[str], Awaitable[None]]
    """Runs before Claude Code starts, given the reserved private Task Queue (for example, to download files from S3)."""
    before_write: Callable[[str], Awaitable[None]]
    """Runs before each write to Claude Code's stdin, given the reserved private Task Queue."""
    on_output: Callable[[str], Awaitable[None]]
    """Runs before the Claude Agent SDK receives each message from Claude Code, given the reserved private Task Queue."""
    after_turn: Callable[[str], Awaitable[None]]
    """Runs after each successful turn, given the reserved private Task Queue (for example, to upload files to S3)."""


class EventPublisher(Protocol):
    """Publishes the messages from Claude Code to a stream.

    The plugin creates one for each Claude Code connection, and calls it in that connection's Activities.
    """

    def before_write(self) -> None:
        """Run at the start of each write Activity, before it writes to Claude Code."""
        ...

    async def publish(self, message: Output) -> None:
        """Publish one message from Claude Code, before the read Activity signals it to the Workflow."""
        ...


@dataclass
class ConnectActivityInput:
    """How to start Claude Code, built from the options as in the Claude Agent SDK."""

    args: list[str]
    """Claude Code's flags."""
    env: dict[str, str]
    """The variables to add to Claude Code's environment."""
    cli_path: str | None
    """``options.cli_path``, or ``None`` so the connector finds Claude Code as in the Claude Agent SDK."""
    cwd: str | None = None
    """``options.cwd``."""
    user: str | None = None
    """``options.user``, the user to run Claude Code as."""
    max_buffer_size: int | None = None
    """``options.max_buffer_size``, the most a message from Claude Code can be."""


@dataclass
class WriteActivityInput:
    """The write Activity's input."""

    text: str
    """One or more lines for Claude Code's stdin."""


@dataclass
class ReadActivityInput:
    """The read Activity's input."""

    signal_name: str
    """The Signal the read Activity sends each message from Claude Code on."""


class Connector(Protocol):
    """Starts Claude Code and reaches it, wherever it runs.

    Exceptions from reading the process terminate the read Activity, which
    Temporal doesn't retry, and the :class:`TemporalTransport`'s ``read_messages`` raises them.

    The returned transport's ``close()`` raises ``ApplicationError(non_retryable=True)``
    when closing fails for good. Temporal retries any other exception, up to the close
    Activity's attempts, so ``close()`` must be safe to call again.
    """

    async def connect(self, request: ConnectActivityInput) -> Transport:
        """Start Claude Code and return a connected Claude Agent SDK ``Transport`` to it.

        The plugin holds the returned transport until the ``close`` Activity,
        and runs every Activity that uses it on this private Worker Entity.
        """
        ...
