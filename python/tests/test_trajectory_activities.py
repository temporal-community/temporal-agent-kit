from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import timedelta

import pytest
from temporal_agent_kit.agent_client import AgentClient
from temporal_agent_kit.agent_protocol import (
    AgentConfig,
    MessageDisposition,
    MessageHandlerEnd,
    TurnEnded,
)
from temporal_agent_kit.evals import Trajectory
from temporal_agent_kit.subagent_activities import SubagentActivities
from temporal_agent_kit.trajectory_activities import (
    AgentTurn,
    CapturedTurn,
    TrajectoryActivities,
)
from temporal_agent_kit.utils.large_payload import local_payload_storage
from temporalio import workflow
from temporalio.api.common.v1 import Payload
from temporalio.client import Client, WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.converter import PayloadCodec
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

_QUEUE = "trajectory-tests"


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
    async def run(self, agent_turn: AgentTurn) -> CapturedTurn:
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
        async with Worker(
            env.client,
            task_queue=_QUEUE,
            workflows=[ScriptedAgent, CaptureTrajectory],
            activities=[
                TrajectoryActivities(env.client).capture_trajectory,
                SubagentActivities(env.client).run_subagent_turn,
            ],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            yield env.client


async def _start_agent(client: Client, workflow_id: str) -> tuple[AgentClient, str]:
    handle = await client.start_workflow(
        ScriptedAgent.run, AgentConfig(), id=workflow_id, task_queue=_QUEUE
    )
    assert handle.result_run_id is not None
    return AgentClient(client, handle.id, run_id=handle.result_run_id), handle.result_run_id


async def _until_completed(agent: AgentClient, turn_number: int) -> None:
    async with asyncio.timeout(10):
        while True:
            status = await agent.get_status()
            if turn_number < status.current_turn or (
                turn_number == status.current_turn and not status.turn_active
            ):
                return
            await asyncio.sleep(0.01)


async def _capture(client: Client, turn: AgentTurn, workflow_id: str) -> CapturedTurn:
    return await client.execute_workflow(
        CaptureTrajectory.run, turn, id=workflow_id, task_queue=_QUEUE
    )


def _capture_error(exc: pytest.ExceptionInfo[WorkflowFailureError]) -> ApplicationError:
    assert isinstance(exc.value.cause, ActivityError)
    error = exc.value.cause.cause
    assert isinstance(error, ApplicationError)
    return error


async def test_capture_rejects_a_turn_that_has_not_completed(worker):
    client = worker
    agent, run_id = await _start_agent(client, "agent")
    script = Script(steps=[ToolStep(id="callback", name="ask_client", args={"question": "?"})])
    held = await agent.submit_message("run_enqueue", script.model_dump(mode="json"))
    async with asyncio.timeout(5):
        while not await agent.get_pending_callbacks():
            await asyncio.sleep(0.01)

    not_completed = [(held.turn_number, "running"), (held.turn_number + 1, "absent")]
    for turn_number, capture_id in not_completed:
        turn = AgentTurn(workflow_id="agent", run_id=run_id, turn_number=turn_number)
        with pytest.raises(WorkflowFailureError) as exc:
            await _capture(client, turn, capture_id)
        error = _capture_error(exc)
        assert error.type == "TurnNotCompleted"
        assert error.non_retryable


async def test_capture_ends_at_the_requested_turn(worker):
    client = worker
    agent, run_id = await _start_agent(client, "agent")
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
    await _until_completed(agent, later.turn_number)

    turn = AgentTurn(workflow_id="agent", run_id=run_id, turn_number=target.turn_number)
    capture = await _capture(client, turn, "capture")

    assert {event.turn_id for event in capture.events} == {"", earlier.turn_id, target.turn_id}
    assert isinstance(capture.events[-1].event, TurnEnded)
    assert capture.events[-1].turn_id == target.turn_id
    trajectory = Trajectory.from_capture(capture)
    assert trajectory.last_turn.reply_text == "target" * 250
    assert {event.turn_id for event in trajectory.last_turn.all} == {target.turn_id}


async def test_capture_reads_the_pinned_run_after_the_workflow_id_is_reused(worker):
    client = worker
    first, first_run = await _start_agent(client, "agent")
    old = Script(steps=[ModelStep(reply=["first run"])])
    accepted = await first.submit_message("run_enqueue", old.model_dump(mode="json"))
    await _until_completed(first, accepted.turn_number)
    await client.get_workflow_handle("agent", run_id=first_run).terminate()

    second, _ = await _start_agent(client, "agent")
    new = Script(steps=[ModelStep(reply=["second run"])])
    reused = await second.submit_message("run_enqueue", new.model_dump(mode="json"))
    await _until_completed(second, reused.turn_number)
    assert reused.turn_number == accepted.turn_number

    turn = AgentTurn(workflow_id="agent", run_id=first_run, turn_number=accepted.turn_number)
    capture = await _capture(client, turn, "historical-capture")
    assert Trajectory.from_capture(capture).last_turn.reply_text == "first run"


async def test_capture_includes_the_subagent_tree(worker):
    client = worker
    agent, run_id = await _start_agent(client, "agent")
    script = Script(
        steps=[
            SubagentStartStep(name="child"),
            SubagentSendStep(name="child", script=Script(steps=[ModelStep(reply=["from child"])])),
            ModelStep(reply=["from root"]),
        ]
    )
    accepted = await agent.submit_message("run_enqueue", script.model_dump(mode="json"))
    await _until_completed(agent, accepted.turn_number)

    turn = AgentTurn(workflow_id="agent", run_id=run_id, turn_number=accepted.turn_number)
    capture = await _capture(client, turn, "tree-capture")

    def reply_texts(events: list) -> dict[str, str]:
        return {
            item.agent_id: item.event.output["text"]
            for item in events
            if isinstance(item.event, MessageHandlerEnd)
        }

    trajectory = Trajectory.from_capture(capture)
    child_ids = {item.agent_id for item in capture.events} - {capture.root_agent_id}
    assert len(child_ids) == 1
    [child] = child_ids
    assert reply_texts(trajectory.last_turn.all) == {
        capture.root_agent_id: "from root",
        child: "from child",
    }
    root_only = Trajectory.from_capture(capture.root_only())
    assert reply_texts(root_only.last_turn.all) == {capture.root_agent_id: "from root"}
