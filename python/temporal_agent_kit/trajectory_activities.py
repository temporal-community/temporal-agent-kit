"""Capture a completed agent turn's retained trajectory for evals."""

from __future__ import annotations

from contextlib import aclosing

from pydantic import BaseModel, ConfigDict, Field
from temporalio import activity
from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.exceptions import ApplicationError

from temporal_agent_kit.agent_protocol import (
    AGENT_STATUS_QUERY,
    AgentEvent,
    AgentStatus,
    SubagentMessageSent,
    SubagentReplyReceived,
    TurnEnded,
    TurnStarted,
)
from temporal_agent_kit.stream_merge import merge_stream, select_replay
from temporal_agent_kit.stream_merge.cursor import Cursor

__all__ = ["CAPTURE_TRAJECTORY_ACTIVITY", "AgentTurn", "TrajectoryActivities"]

CAPTURE_TRAJECTORY_ACTIVITY = "agentkit.eval.capture_trajectory"


class AgentTurn(BaseModel):
    """The source Workflow and turn requested for evaluation."""

    model_config = ConfigDict(frozen=True)

    workflow_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    turn_number: int = Field(ge=1)


class TrajectoryActivities:
    """Bind trajectory capture to the Temporal client that reads the source stream."""

    def __init__(self, client: Client) -> None:
        self._client = client

    @activity.defn(name=CAPTURE_TRAJECTORY_ACTIVITY)
    async def capture_trajectory(self, agent_turn: AgentTurn) -> list[AgentEvent]:
        """Read retained agent and subagent events through a completed turn."""
        handle = self._client.get_workflow_handle(agent_turn.workflow_id, run_id=agent_turn.run_id)
        status = await handle.query(AGENT_STATUS_QUERY, result_type=AgentStatus)
        turn_number = agent_turn.turn_number
        if turn_number > status.current_turn or (
            turn_number == status.current_turn and status.turn_active
        ):
            raise ApplicationError(
                f"Turn {turn_number} has not completed "
                f"(current turn {status.current_turn}, active: {status.turn_active})",
                type="TurnNotCompleted",
                non_retryable=True,
            )

        head = await WorkflowStreamClient(handle).get_offset()
        events: list[AgentEvent] = []
        target_end: AgentEvent | None = None

        async def should_stop(cursor: Cursor, event: AgentEvent) -> bool:
            nonlocal target_end
            if cursor.is_child:
                return False
            if event.turn_number == turn_number and isinstance(event.event, TurnEnded):
                target_end = event
                return True
            # A completed turn missing from the retained stream cannot appear in its live tail.
            return cursor.head_offset + 1 >= head

        if head:
            merged = merge_stream(
                client=self._client,
                root_workflow_id=agent_turn.workflow_id,
                root_run_id=agent_turn.run_id,
                root_from_offset=0,
                skip_until_turn_id=None,
                select=select_replay,
                should_stop=should_stop,
            )
            async with aclosing(merged) as items:
                async for event, _ in items:
                    events.append(event)

        if target_end is None:
            raise ApplicationError(
                f"The retained workflow stream is missing the end of turn {turn_number}",
                type="IncompleteTrajectory",
                non_retryable=True,
            )
        parent_id = target_end.agent_id
        if not any(
            item.agent_id == parent_id
            and item.turn_id == target_end.turn_id
            and isinstance(item.event, TurnStarted)
            for item in events
        ):
            raise ApplicationError(
                "The requested turn's start is outside the retained stream",
                type="IncompleteTrajectory",
                non_retryable=True,
            )
        # Queued parent turns may already have admission events. A child's turn
        # number belongs to its own stream and is unrelated to the parent's.
        captured = [
            item
            for item in events
            if item.agent_id != parent_id or item.turn_number <= target_end.turn_number
        ]
        _validate_subagent_turns(captured)
        return captured


def _validate_subagent_turns(events: list[AgentEvent]) -> None:
    # A UI can continue when a child stream is missing. Evaluators need to know
    # when a retained dispatch or reply has an incomplete child trajectory.
    expected = {
        (item.event.subagent_id, item.event.subagent_turn)
        for item in events
        if isinstance(item.event, (SubagentMessageSent, SubagentReplyReceived))
    }
    if not expected:
        return
    started = {
        (item.agent_id, item.turn_number) for item in events if isinstance(item.event, TurnStarted)
    }
    ended = {
        (item.agent_id, item.turn_number) for item in events if isinstance(item.event, TurnEnded)
    }
    if missing := expected - (started & ended):
        agent_id, turn_number = min(missing)
        raise ApplicationError(
            f"The retained trajectory is missing events for subagent {agent_id!r} turn {turn_number}",
            type="IncompleteTrajectory",
            non_retryable=True,
        )
