"""Capture a completed agent turn's trajectory, including its subagents, for evals."""

from __future__ import annotations

from contextlib import aclosing

from pydantic import BaseModel, ConfigDict, Field
from temporalio import activity
from temporalio.client import Client
from temporalio.exceptions import ApplicationError

from temporal_agent_kit.agent_client import AgentClient, AgentStreamOutput
from temporal_agent_kit.agent_protocol import (
    AgentEvent,
    AgentStatus,
    SubagentStreamUnavailable,
    TurnEnded,
    TurnStarted,
)

__all__ = ["CAPTURE_TRAJECTORY_ACTIVITY", "AgentTurn", "CapturedTurn", "TrajectoryActivities"]

CAPTURE_TRAJECTORY_ACTIVITY = "agentkit.eval.capture_trajectory"


class AgentTurn(BaseModel):
    """One turn of one exact agent run. The run ID matters because workflow IDs can be reused."""

    model_config = ConfigDict(frozen=True)

    workflow_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    turn_number: int = Field(ge=1)


class CapturedTurn(BaseModel):
    """The merged agent-tree events through the end of ``source``'s turn, in merged order."""

    model_config = ConfigDict(frozen=True)

    source: AgentTurn
    root_agent_id: str
    events: list[AgentEvent]

    def root_only(self) -> CapturedTurn:
        """The same capture without any subagent's events."""
        return self.model_copy(
            update={"events": [e for e in self.events if e.agent_id == self.root_agent_id]}
        )


def _turn_completed(status: AgentStatus, turn_number: int) -> bool:
    if turn_number < status.current_turn:
        return True
    return turn_number == status.current_turn and not status.turn_active


class TrajectoryActivities:
    """Bind trajectory capture to the Temporal client that reads the agent tree's streams."""

    def __init__(self, client: Client) -> None:
        self._client = client

    @activity.defn(name=CAPTURE_TRAJECTORY_ACTIVITY)
    async def capture_trajectory(self, source: AgentTurn) -> CapturedTurn:
        """Read the agent tree's events through the end of a turn that has already completed.

        Every read is pinned to ``source.run_id``, so a reused workflow ID never substitutes
        another run. A turn that has not completed is rejected before any stream is read. Every subagent turn
        nested in a completed turn has completed too, because a parent publishes a subagent's
        reply only after that subagent's turn ends. An unreadable subagent stream fails the
        capture instead of producing a partial tree.
        """
        agent = AgentClient(self._client, source.workflow_id, run_id=source.run_id)
        status = await agent.get_status()
        if not _turn_completed(status, source.turn_number):
            raise ApplicationError(
                f"Turn {source.turn_number} has not completed "
                f"(current turn {status.current_turn}, active: {status.turn_active})",
                type="TurnNotCompleted",
                non_retryable=True,
            )
        root = status.agent_id
        events: list[AgentEvent] = []
        started = False

        def keep(item: AgentStreamOutput, _offset: int) -> AgentStreamOutput:
            return item

        async with aclosing(await agent.attach(on_item=keep)) as stream:
            async for item in stream:
                if not isinstance(item, AgentEvent):
                    raise ApplicationError(
                        f"Unexpected stream item {type(item).__name__}",
                        type="IncompleteTrajectory",
                        non_retryable=True,
                    )
                if isinstance(item.event, SubagentStreamUnavailable):
                    raise ApplicationError(
                        f"Subagent {item.event.subagent_id} stream is unavailable: "
                        f"{item.event.reason}",
                        type="IncompleteTrajectory",
                        non_retryable=True,
                    )
                is_root = item.agent_id == root
                # A queued turn's admission can land on the root stream before this turn ends.
                if is_root and item.turn_number > source.turn_number:
                    continue
                events.append(item)
                if not is_root or item.turn_number != source.turn_number:
                    continue
                if isinstance(item.event, TurnStarted):
                    started = True
                elif isinstance(item.event, TurnEnded):
                    if not started:
                        break
                    return CapturedTurn(source=source, root_agent_id=root, events=events)

        raise ApplicationError(
            f"The stream does not contain all of turn {source.turn_number}",
            type="IncompleteTrajectory",
            non_retryable=True,
        )
