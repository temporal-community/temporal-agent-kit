from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest
from temporal_agent_kit import AgentEvents, EvaluatorInput, trajectory_activities
from temporal_agent_kit.agent_client import AgentClient
from temporal_agent_kit.agent_protocol import (
    AgentConfig,
    AgentEvent,
    MessageDisposition,
    SubagentMessageSent,
    ToolErrorEvent,
    TurnEnded,
    TurnStarted,
)
from temporal_agent_kit.subagent_activities import SubagentActivities
from temporal_agent_kit.trajectory_activities import AgentTurn, TrajectoryActivities
from temporalio import workflow
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from tests.conformance.script import (
    ModelStep,
    Script,
    SubagentSendStep,
    SubagentStartStep,
    ToolStep,
)
from tests.conformance.scripted_agent import ScriptedAgent


@workflow.defn
class CaptureTrajectory:
    @workflow.run
    async def run(self, agent_turn: AgentTurn) -> list[AgentEvent]:
        return await workflow.execute_activity_method(
            TrajectoryActivities.capture_trajectory,
            agent_turn,
            start_to_close_timeout=timedelta(seconds=10),
        )


@pytest.fixture
async def worker():
    async with await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter) as env:
        trajectory = TrajectoryActivities(env.client)
        subagents = SubagentActivities(env.client)
        async with Worker(
            env.client,
            task_queue="trajectory-tests",
            workflows=[ScriptedAgent, CaptureTrajectory],
            activities=[trajectory.capture_trajectory, subagents.run_subagent_turn],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            yield env.client


async def test_capture_returns_native_events_through_the_requested_turn(worker):
    client = worker
    handle = await client.start_workflow(
        ScriptedAgent.run, AgentConfig(), id="agent", task_queue="trajectory-tests"
    )
    assert handle.result_run_id is not None
    agent = AgentClient(client, handle.id)
    earlier = await agent.submit_message("run_enqueue", {"steps": []})
    script = Script(
        steps=[
            ToolStep(id="callback", name="ask_client", args={"question": "Ready?"}),
            ModelStep(reply=["Target turn reply."]),
        ]
    )
    target = await agent.submit_message("run_enqueue", script.model_dump(mode="json"))
    later = await agent.submit_message("run_enqueue", {"steps": []})
    assert later.disposition is MessageDisposition.QUEUED
    async with asyncio.timeout(5):
        while not await agent.get_pending_callbacks():
            await asyncio.sleep(0.01)
    await agent.provide_callback_result("callback", result="yes")
    await _wait_for_completed_turn(agent, target.turn_number)

    turn = AgentTurn(
        workflow_id=handle.id, run_id=handle.result_run_id, turn_number=target.turn_number
    )
    events = await client.execute_workflow(
        CaptureTrajectory.run, turn, id="archive", task_queue="trajectory-tests"
    )

    assert {event.turn_id for event in events} == {"", earlier.turn_id, target.turn_id}
    assert isinstance(events[-1].event, TurnEnded)
    assert events[-1].turn_id == target.turn_id
    assert AgentEvents(events).for_turn(target.turn_id).reply_text == "Target turn reply."


async def test_capture_reads_closed_run_after_workflow_id_is_reused(worker):
    client = worker
    handle = await client.start_workflow(
        ScriptedAgent.run, AgentConfig(), id="agent", task_queue="trajectory-tests"
    )
    assert handle.result_run_id is not None
    agent = AgentClient(client, handle.id)
    original = Script(steps=[ModelStep(reply=["Original run."])])
    accepted = await agent.submit_message("run_enqueue", original.model_dump(mode="json"))
    turn = AgentTurn(
        workflow_id=handle.id, run_id=handle.result_run_id, turn_number=accepted.turn_number
    )
    await _wait_for_completed_turn(agent, accepted.turn_number)
    await handle.terminate()

    await client.start_workflow(
        ScriptedAgent.run, AgentConfig(), id=handle.id, task_queue="trajectory-tests"
    )
    replacement = Script(steps=[ModelStep(reply=["Replacement run."])])
    newer = await agent.submit_message("run_enqueue", replacement.model_dump(mode="json"))
    await _wait_for_completed_turn(agent, newer.turn_number)
    assert newer.turn_number == accepted.turn_number

    events = await client.execute_workflow(
        CaptureTrajectory.run, turn, id="historical-capture", task_queue="trajectory-tests"
    )
    assert isinstance(events[-1].event, TurnEnded)
    assert events[-1].turn_id == accepted.turn_id
    assert AgentEvents(events).for_turn(accepted.turn_id).reply_text == "Original run."


async def test_capture_includes_nested_subagent_turns_after_the_parent_closes(worker):
    client = worker
    handle = await client.start_workflow(
        ScriptedAgent.run, AgentConfig(), id="agent", task_queue="trajectory-tests"
    )
    assert handle.result_run_id is not None
    agent = AgentClient(client, handle.id)
    earlier_script = Script(
        steps=[
            SubagentStartStep(name="child"),
            SubagentSendStep(
                name="child", script=Script(steps=[ModelStep(reply=["Earlier child reply."])])
            ),
        ]
    )
    earlier = await agent.submit_message("run_enqueue", earlier_script.model_dump(mode="json"))
    await _wait_for_completed_turn(agent, earlier.turn_number)

    child_script = Script(steps=[ModelStep(reply=["Child reply."])])
    nested_script = Script(
        steps=[
            SubagentStartStep(name="grandchild"),
            SubagentSendStep(
                name="grandchild",
                script=Script(steps=[ToolStep(id="fail", name="fail", args={"message": "failed"})]),
            ),
        ]
    )
    script = Script(
        steps=[
            SubagentSendStep(name="child", script=child_script),
            SubagentSendStep(name="child", script=nested_script),
            ModelStep(reply=["Parent reply."]),
        ]
    )
    accepted = await agent.submit_message("run_enqueue", script.model_dump(mode="json"))
    turn = AgentTurn(
        workflow_id=handle.id, run_id=handle.result_run_id, turn_number=accepted.turn_number
    )
    await _wait_for_completed_turn(agent, accepted.turn_number)
    await handle.terminate()
    merged = await client.execute_workflow(
        CaptureTrajectory.run, turn, id="merged-capture", task_queue="trajectory-tests"
    )

    events = AgentEvents(merged)
    turn_events = EvaluatorInput(turn_id=accepted.turn_id, events=events).turn_events
    parent = turn_events.parent_events
    assert parent.reply_text == "Parent reply."
    assert parent.all == events.parent_events.for_turn(accepted.turn_id).all
    dispatches = [item.event for item in parent.all if isinstance(item.event, SubagentMessageSent)]
    assert len(dispatches) == 2
    child_id = dispatches[0].subagent_id
    assert [item.turn_number for item in events.for_agent(child_id).of_type(TurnEnded).all] == [
        1, 2, 3
    ]
    assert [item.turn_number for item in turn_events.for_agent(child_id).of_type(TurnEnded).all] == [
        2, 3
    ]
    [failure] = turn_events.of_type(ToolErrorEvent).all
    assert failure.agent_id.startswith(f"{child_id}-")
    assert turn_events.of_type(ToolErrorEvent).parent_events.all == []


async def test_merged_capture_fails_if_a_child_turn_start_is_missing(worker, monkeypatch):
    handle = await worker.start_workflow(
        ScriptedAgent.run, AgentConfig(), id="parent", task_queue="trajectory-tests"
    )
    assert handle.result_run_id is not None
    agent = AgentClient(worker, handle.id)
    accepted = await agent.submit_message("run_enqueue", {"steps": []})
    await _wait_for_completed_turn(agent, accepted.turn_number)

    parent_start = AgentEvent(
        agent_id="a1b2c3", turn_id="parent-turn", turn_number=1, timestamp=0, event=TurnStarted()
    )
    dispatch = parent_start.model_copy(
        update={
            "event": SubagentMessageSent(
                subagent_id="a1b2c3-d4e5f6",
                agent_key="child",
                workflow_id="child",
                handler="chat",
                subagent_turn=2,
            )
        }
    )
    child_end = AgentEvent(
        agent_id="a1b2c3-d4e5f6",
        turn_id="child-turn",
        turn_number=2,
        timestamp=1,
        event=TurnEnded(),
    )
    parent_end = parent_start.model_copy(update={"event": TurnEnded()})

    async def truncated_merge(*, should_stop, **kwargs):
        for offset, event in enumerate([parent_start, dispatch, child_end, parent_end]):
            yield event, offset
            cursor = SimpleNamespace(
                is_child=event.agent_id != parent_start.agent_id, head_offset=offset
            )
            if await should_stop(cursor, event):
                return

    monkeypatch.setattr(trajectory_activities, "merge_stream", truncated_merge)
    turn = AgentTurn(
        workflow_id=handle.id, run_id=handle.result_run_id, turn_number=accepted.turn_number
    )
    with pytest.raises(ApplicationError, match="missing events for subagent 'a1b2c3-d4e5f6' turn 2") as exc:
        await TrajectoryActivities(worker).capture_trajectory(turn)

    assert exc.value.type == "IncompleteTrajectory"
    assert exc.value.non_retryable


async def test_capture_rejects_active_and_future_turns(worker, monkeypatch):
    client = worker
    handle = await client.start_workflow(
        ScriptedAgent.run, AgentConfig(), id="agent", task_queue="trajectory-tests"
    )
    assert handle.result_run_id is not None
    agent = AgentClient(client, handle.id)
    script = Script(
        steps=[ToolStep(id="hold", name="ask_client", args={"question": "Ready?"})]
    )
    held = await agent.submit_message("run_enqueue", script.model_dump(mode="json"))
    async with asyncio.timeout(5):
        while not await agent.get_pending_callbacks():
            await asyncio.sleep(0.01)

    def unexpected_merge(**_):
        raise AssertionError("Capture must check turn completion before reading events")

    monkeypatch.setattr(trajectory_activities, "merge_stream", unexpected_merge)
    for turn_number in (held.turn_number, held.turn_number + 1):
        turn = AgentTurn(
            workflow_id=handle.id, run_id=handle.result_run_id, turn_number=turn_number
        )
        with pytest.raises(WorkflowFailureError) as exc:
            await client.execute_workflow(
                CaptureTrajectory.run,
                turn,
                id=f"unfinished-turn-{turn_number}",
                task_queue="trajectory-tests",
                execution_timeout=timedelta(seconds=10),
            )

        assert isinstance(exc.value.cause, ActivityError)
        error = exc.value.cause.cause
        assert isinstance(error, ApplicationError)
        assert error.type == "TurnNotCompleted"
        assert error.non_retryable
        assert f"Turn {turn_number} has not completed" in error.message


async def _wait_for_completed_turn(agent: AgentClient, turn_number: int) -> None:
    async with asyncio.timeout(5):
        while True:
            status = await agent.get_status()
            if status.current_turn > turn_number or (
                status.current_turn == turn_number and not status.turn_active
            ):
                return
            await asyncio.sleep(0.01)
