from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import timedelta

import pytest
from temporal_agent_kit import AgentEvents
from temporal_agent_kit.agent_client import AgentClient
from temporal_agent_kit.agent_protocol import (
    AgentConfig,
    AgentEvent,
    MessageDisposition,
    TurnEnded,
)
from temporal_agent_kit.trajectory_activities import AgentTurn, TrajectoryActivities
from temporal_agent_kit.utils.large_payload import local_payload_storage
from temporalio import workflow
from temporalio.api.common.v1 import Payload
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.converter import PayloadCodec
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from tests.conformance.script import ModelStep, Script, ToolStep
from tests.conformance.scripted_agent import ScriptedAgent


class TestPayloadCodec(PayloadCodec):
    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload(metadata={"encoding": b"binary/test"}, data=item.SerializeToString()[::-1])
            for item in payloads
        ]

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [Payload.FromString(item.data[::-1]) for item in payloads]


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
async def worker(tmp_path):
    converter = replace(
        pydantic_data_converter,
        payload_codec=TestPayloadCodec(),
        external_storage=local_payload_storage(
            base_dir=tmp_path / "automatic", payload_size_threshold=1_000
        ),
    )
    async with await WorkflowEnvironment.start_local(data_converter=converter) as env:
        trajectory = TrajectoryActivities(env.client)
        async with Worker(
            env.client,
            task_queue="trajectory-tests",
            workflows=[ScriptedAgent, CaptureTrajectory],
            activities=[trajectory.capture_trajectory],
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
            ModelStep(reply=["target" * 250]),
        ]
    )
    target = await agent.submit_message("run_enqueue", script.model_dump(mode="json"))
    later = await agent.submit_message("run_enqueue", {"steps": []})
    assert later.disposition is MessageDisposition.QUEUED
    async with asyncio.timeout(5):
        while not await agent.get_pending_callbacks():
            await asyncio.sleep(0.01)
    await agent.provide_callback_result("callback", result="yes")

    turn = AgentTurn(workflow_id=handle.id, run_id=handle.result_run_id, turn_id=target.turn_id)
    events = await client.execute_workflow(
        CaptureTrajectory.run, turn, id="archive", task_queue="trajectory-tests"
    )

    assert {event.turn_id for event in events} == {"", earlier.turn_id, turn.turn_id}
    assert isinstance(events[-1].event, TurnEnded)
    assert events[-1].turn_id == turn.turn_id
    assert AgentEvents(events).for_turn(turn.turn_id).reply_text == "target" * 250


async def test_capture_pins_the_run_even_after_the_workflow_id_is_reused(worker):
    client = worker
    handle = await client.start_workflow(
        ScriptedAgent.run, AgentConfig(), id="agent", task_queue="trajectory-tests"
    )
    assert handle.result_run_id is not None
    accepted = await AgentClient(client, handle.id).submit_message("run_enqueue", {"steps": []})
    turn = AgentTurn(workflow_id=handle.id, run_id=handle.result_run_id, turn_id=accepted.turn_id)
    await client.execute_workflow(
        CaptureTrajectory.run, turn, id="first-capture", task_queue="trajectory-tests"
    )
    await handle.terminate()
    await client.start_workflow(
        ScriptedAgent.run, AgentConfig(), id=handle.id, task_queue="trajectory-tests"
    )

    events = await client.execute_workflow(
        CaptureTrajectory.run, turn, id="historical-capture", task_queue="trajectory-tests"
    )
    assert events[-1].turn_id == turn.turn_id

    with pytest.raises(WorkflowFailureError) as exc:
        await client.execute_workflow(
            CaptureTrajectory.run,
            turn.model_copy(update={"turn_id": "absent"}),
            id="missing-turn",
            task_queue="trajectory-tests",
        )
    assert isinstance(exc.value.cause, ActivityError)
    error = exc.value.cause.cause
    assert isinstance(error, ApplicationError)
    assert error.type == "IncompleteTrajectory"
    assert error.non_retryable
