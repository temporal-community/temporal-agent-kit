"""Capture a completed agent turn's retained trajectory for evals."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from datetime import timedelta

from pydantic import BaseModel, ConfigDict, Field
from temporalio import activity
from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.exceptions import ApplicationError

from temporal_agent_kit.agent_protocol import (
    TURN_EVENTS_TOPIC,
    AgentEvent,
    TurnEnded,
    TurnStarted,
)

__all__ = ["CAPTURE_TRAJECTORY_ACTIVITY", "AgentTurn", "TrajectoryActivities"]

CAPTURE_TRAJECTORY_ACTIVITY = "agentkit.eval.capture_trajectory"


class AgentTurn(BaseModel):
    """The source Workflow and turn requested for evaluation."""

    model_config = ConfigDict(frozen=True)

    workflow_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)


class TrajectoryActivities:
    """Bind trajectory capture to the Temporal client that reads the source stream."""

    def __init__(
        self,
        client: Client,
        *,
        stream_timeout: timedelta = timedelta(seconds=30),
    ) -> None:
        self._client = client
        self._stream_timeout = stream_timeout

    @activity.defn(name=CAPTURE_TRAJECTORY_ACTIVITY)
    async def capture_trajectory(self, agent_turn: AgentTurn) -> list[AgentEvent]:
        """Read the source agent's retained events through the requested completed turn."""
        stream = WorkflowStreamClient.create(self._client, agent_turn.workflow_id)
        subscription = stream.subscribe(TURN_EVENTS_TOPIC, result_type=AgentEvent)
        events: list[AgentEvent] = []
        started = False
        timeout_seconds = self._stream_timeout.total_seconds()
        deadline = asyncio.timeout(timeout_seconds)
        try:
            async with deadline, aclosing(subscription) as items:
                async for item in items:
                    event = item.data
                    events.append(event)
                    if event.turn_id != agent_turn.turn_id:
                        continue
                    if isinstance(event.event, TurnStarted):
                        started = True
                    if isinstance(event.event, TurnEnded):
                        if not started:
                            raise ApplicationError(
                                "The requested turn's start is outside the retained stream",
                                type="IncompleteTrajectory",
                                non_retryable=True,
                            )
                        # Admission for a queued turn can appear before this turn ends.
                        return [item for item in events if item.turn_number <= event.turn_number]
        except TimeoutError:
            if not deadline.expired():
                raise
            raise ApplicationError(
                f"The requested turn {agent_turn.turn_id!r} did not complete within "
                f"{timeout_seconds:g} seconds of trajectory capture",
                type="IncompleteTrajectory",
                non_retryable=True,
            ) from None

        raise ApplicationError(
            "The workflow stream ended before the requested turn completed",
            type="IncompleteTrajectory",
            non_retryable=True,
        )
