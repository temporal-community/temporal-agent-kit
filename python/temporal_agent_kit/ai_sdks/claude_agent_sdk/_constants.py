"""Constants shared across the Claude Agent SDK integration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Final

from temporalio import workflow
from temporalio.common import RetryPolicy

STDIN_LABELS: Final[Mapping[str, str]] = {
    "user": "prompt",
    "control_request": "request",
    "control_response": "reply",
}

CONNECT: Final = "claude-agent-sdk-connect"
WRITE: Final = "claude-agent-sdk-write"
END_INPUT: Final = "claude-agent-sdk-end-input"
READ: Final = "claude-agent-sdk-read"
CLOSE: Final = "claude-agent-sdk-close"
RESERVE_PRIVATE_TASK_QUEUE: Final = "claude-agent-sdk-reserve-private-task-queue"

MAX_WORKER_CONNECT_ATTEMPTS: Final = 3
"""How many times ``TemporalTransport.connect()`` tries to start Claude Code, 30 seconds apart."""

DEFAULT_MAX_CONNECTIONS: Final = 10
"""The most Claude Code connections one private Worker Entity runs at once, unless the plugin
sets it."""

# Default options for each Activity, which the user's activity options override.

RESERVE_PRIVATE_TASK_QUEUE_OPTIONS: Final[workflow.ActivityConfig] = {
    "summary": "Reserve Claude Code connection",
    "schedule_to_close_timeout": timedelta(seconds=30),
    "start_to_close_timeout": timedelta(seconds=10),
    "retry_policy": RetryPolicy(maximum_attempts=1),
}

CONNECT_OPTIONS: Final[workflow.ActivityConfig] = {
    "summary": "Start Claude Code",
    "start_to_close_timeout": timedelta(minutes=5),
    "heartbeat_timeout": timedelta(seconds=10),
    "schedule_to_start_timeout": timedelta(seconds=10),
    "retry_policy": RetryPolicy(maximum_attempts=3)
}

WRITE_OPTIONS: Final[workflow.ActivityConfig] = {
    "start_to_close_timeout": timedelta(minutes=5),
    "heartbeat_timeout": timedelta(seconds=10),
    "schedule_to_start_timeout": timedelta(seconds=10),
    "retry_policy": RetryPolicy(maximum_attempts=3)
}

READ_OPTIONS: Final[workflow.ActivityConfig] = {
    "summary": "Read output",
    "start_to_close_timeout": timedelta(hours=8),
    "heartbeat_timeout": timedelta(seconds=10),
    "schedule_to_start_timeout": timedelta(seconds=10),
    "retry_policy": RetryPolicy(maximum_attempts=1),
}

END_INPUT_OPTIONS: Final[workflow.ActivityConfig] = {
    "summary": "End input",
    "start_to_close_timeout": timedelta(minutes=5),
    "heartbeat_timeout": timedelta(seconds=10),
    "schedule_to_start_timeout": timedelta(seconds=10),
    "retry_policy": RetryPolicy(maximum_attempts=3)
}

CLOSE_OPTIONS: Final[workflow.ActivityConfig] = {
    "summary": "Close Claude Code",
    "start_to_close_timeout": timedelta(minutes=5),
    "heartbeat_timeout": timedelta(seconds=10),
    "schedule_to_start_timeout": timedelta(seconds=10),
    "retry_policy": RetryPolicy(maximum_attempts=3)
}
