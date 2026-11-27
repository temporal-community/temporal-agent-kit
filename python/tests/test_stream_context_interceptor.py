"""Tests that an Activity sees the stream context its Workflow set when it started the Activity."""

from __future__ import annotations

import uuid
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from temporal_agent_kit.agent_workflow import _CURRENT_STREAM_CONTEXT
from temporal_agent_kit.plugin import AgentKitPlugin
from temporal_agent_kit.stream_context import TurnStreamContext

CONTEXT = TurnStreamContext(agent_id="agent", turn_id="turn", turn_number=1, message_id="message")


@activity.defn
async def read_stream_context() -> TurnStreamContext | None:
    return _CURRENT_STREAM_CONTEXT.get()


@workflow.defn(sandboxed=False)
class SetStreamContextWorkflow:
    @workflow.run
    async def run(self) -> TurnStreamContext | None:
        _CURRENT_STREAM_CONTEXT.set(CONTEXT)
        return await workflow.execute_activity(read_stream_context, start_to_close_timeout=timedelta(seconds=10))


async def test_activity_gets_the_stream_context_the_workflow_set() -> None:
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        task_queue = f"stream-context-{uuid.uuid4()}"
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[SetStreamContextWorkflow],
            activities=[read_stream_context],
            workflow_runner=UnsandboxedWorkflowRunner(),
            plugins=[AgentKitPlugin()],
        ):
            result = await env.client.execute_workflow(
                SetStreamContextWorkflow.run,
                id=f"stream-context-{uuid.uuid4()}",
                task_queue=task_queue,
                result_type=TurnStreamContext,
            )

    assert result == CONTEXT
