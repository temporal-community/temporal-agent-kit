from __future__ import annotations

from collections.abc import Sequence

import pytest
from temporalio import activity, workflow
from temporalio.api.common.v1 import Payload
from temporalio.api.sdk.v1.external_storage_pb2 import ExternalStorageReference
from temporalio.converter import (
    DataConverter,
    ExternalStorage,
    JSONProtoPayloadConverter,
    PayloadCodec,
)
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker

with workflow.unsafe.imports_passed_through():
    from examples.openai_hello.evaluators import count_failed_tools, has_reply
    from temporal_agent_kit import AgentEvents, agent
    from temporal_agent_kit.agent_client import AgentClient
    from temporal_agent_kit.agent_protocol import (
        AgentConfig,
        AgentEvent,
        ToolErrorEvent,
    )
    from temporal_agent_kit.ai_sdks.openai_agents import OpenAIAgentsPlugin
    from temporal_agent_kit.eval_workflow import RunEvaluatorsWorkflow
    from temporal_agent_kit.evaluators import EvaluatorResult
    from temporal_agent_kit.plugin import AgentKitPlugin
    from temporal_agent_kit.trajectory_activities import AgentTurn
    from temporal_agent_kit.utils.large_payload import LocalFileStorageDriver
    from tests.conformance.script import ModelStep, Script, ToolStep
    from tests.conformance.scripted_agent import SCRIPTED_AGENT, ScriptedAgent
    from tests.plugin.inline_evaluator_fixture import (
        invalid_inline_result,
        unavailable_inline_evaluator,
    )


class _WrappingCodec(PayloadCodec):
    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload(metadata={"encoding": b"binary/test"}, data=item.SerializeToString()[::-1])
            for item in payloads
        ]

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        assert all(item.metadata == {"encoding": b"binary/test"} for item in payloads)
        return [Payload.FromString(item.data[::-1]) for item in payloads]


@agent.activity_eval_defn()
async def prior_failed_tools(events: AgentEvents) -> EvaluatorResult:
    turn = events.for_turn(events.all[-1].turn_id)
    if len(events.replies) == len(turn.replies):
        return EvaluatorResult(status="skipped", reasoning="No earlier replies.")
    all_failures = len(events.of_type(ToolErrorEvent).all)
    target_failures = len(turn.of_type(ToolErrorEvent).all)
    return EvaluatorResult(status="completed", value=all_failures - target_failures)


@agent.activity_eval_defn
async def unavailable_judge(events: AgentEvents) -> EvaluatorResult:
    raise ApplicationError("Judge unavailable.", non_retryable=True)


@agent.activity_eval_defn
async def other_agents_eval(events: AgentEvents) -> EvaluatorResult:
    return EvaluatorResult(status="completed", value=42)


def test_plugin_validates_explicit_evaluator_registration():
    async def undecorated(_):
        return 0

    with pytest.raises(TypeError, match="requires @agent.eval_defn or @agent.activity_eval_defn"):
        AgentKitPlugin(evaluators=[undecorated])
    with pytest.raises(ValueError, match="duplicate evaluator name"):
        AgentKitPlugin(evaluators=[count_failed_tools, count_failed_tools])


@pytest.mark.parametrize("codec", [None, _WrappingCodec()])
@pytest.mark.parametrize(
    ("threshold", "offloaded"),
    [(3_000, True), (1_000_000, False)],
    ids=["above-threshold", "below-threshold"],
)
async def test_runner_uses_configured_storage_and_keeps_evaluator_outcomes_independent(
    tmp_path, codec, threshold, offloaded
):
    unused = LocalFileStorageDriver(tmp_path / "unused")
    archive = LocalFileStorageDriver(tmp_path / "archive", driver_name="archive")
    converter = DataConverter(
        payload_codec=codec,
        external_storage=ExternalStorage(
            drivers=[unused, archive],
            driver_selector=lambda context, payload: archive,
            payload_size_threshold=threshold,
        ),
    )
    plugin = AgentKitPlugin(
        evaluators={
            SCRIPTED_AGENT: [
                count_failed_tools,
                has_reply,
                unavailable_inline_evaluator,
                prior_failed_tools,
                invalid_inline_result,
                unavailable_judge,
            ],
            "OtherAgent": [other_agents_eval],
        }
    )
    async with await WorkflowEnvironment.start_local(
        plugins=[OpenAIAgentsPlugin(register_activities=False), plugin],
        data_converter=converter,
    ) as env:
        client = env.client

        @activity.defn(name="capture_trajectory")
        async def application_capture() -> None:
            raise AssertionError("Expected the Kit's capture Activity")

        async with (
            Worker(
                client,
                task_queue="eval-source",
                workflows=[ScriptedAgent],
                workflow_runner=UnsandboxedWorkflowRunner(),
            ),
            Worker(client, task_queue="evals", activities=[application_capture]),
        ):
            handle = await client.start_workflow(
                ScriptedAgent.run, AgentConfig(), id="eval-agent", task_queue="eval-source"
            )
            assert handle.result_run_id is not None
            source = AgentClient(client, handle.id)
            source_text = "A captured source reply. " * 200
            script = Script(
                steps=[
                    ToolStep(id="failing-tool", name="fail", args={"message": "failed"}),
                    ModelStep(reply=[source_text]),
                ],
            )
            failed = await source.submit_message("run_enqueue", script.model_dump(mode="json"))
            clean = await source.submit_message("run_enqueue", {"steps": []})
            first_turn = AgentTurn(
                workflow_id=handle.id, run_id=handle.result_run_id, turn_id=failed.turn_id
            )
            failed_eval = await client.start_workflow(
                RunEvaluatorsWorkflow.run, first_turn, id="eval-first", task_queue="evals"
            )
            clean_eval = await client.start_workflow(
                RunEvaluatorsWorkflow.run,
                first_turn.model_copy(update={"turn_id": clean.turn_id}),
                id="eval-second",
                task_queue="evals",
            )
            failed_result = await failed_eval.result()
            clean_result = await clean_eval.result()
            history = await clean_eval.fetch_history()
        await Replayer(
            workflows=[RunEvaluatorsWorkflow], data_converter=client.data_converter
        ).replay_workflow(history)

    assert failed_result.schema_version == 1
    assert failed_result.agent == clean_result.agent == SCRIPTED_AGENT
    assert failed_result.source == first_turn
    assert clean_result.source.turn_id == clean.turn_id
    assert set(clean_result.results) == {
        "count_failed_tools",
        "has_reply",
        "invalid_inline_result",
        "prior_failed_tools",
        "unavailable_inline_evaluator",
        "unavailable_judge",
    }
    assert failed_result.results["count_failed_tools"].value == 1
    assert clean_result.results["count_failed_tools"].value == 0
    assert failed_result.results["has_reply"].value == 1
    assert clean_result.results["has_reply"].value == 0
    missing = clean_result.results["unavailable_inline_evaluator"]
    assert missing == EvaluatorResult(
        status="error",
        reasoning="Inline evaluator tests.plugin.inline_evaluator_fixture.unavailable_inline_evaluator is unavailable",
    )
    invalid = clean_result.results["invalid_inline_result"]
    assert invalid.status == "error"
    assert invalid.value is None
    assert invalid.reasoning is not None and "require reasoning and no score" in invalid.reasoning
    previous = failed_result.results["prior_failed_tools"]
    assert previous.status == "skipped"
    assert previous.reasoning == "No earlier replies."
    assert previous.value is None
    assert clean_result.results["prior_failed_tools"].value == 1
    error = clean_result.results["unavailable_judge"]
    assert error.status == "error"
    assert error.value is None
    assert error.reasoning is not None and "Judge unavailable." in error.reasoning

    scheduled = {
        event.event_id: event.activity_task_scheduled_event_attributes
        for event in history.events
        if event.HasField("activity_task_scheduled_event_attributes")
    }
    [capture_id] = [
        event_id
        for event_id, event in scheduled.items()
        if event.activity_type.name == "agentkit.eval.capture_trajectory"
    ]
    completed_activities = [
        event.activity_task_completed_event_attributes
        for event in history.events
        if event.HasField("activity_task_completed_event_attributes")
    ]
    activity_results = {event.scheduled_event_id: event.result for event in completed_activities}
    [capture_payload] = activity_results[capture_id].payloads

    evaluator_inputs = [
        event.input.payloads[0]
        for event_id, event in scheduled.items()
        if event.activity_type.name.startswith("agentkit.eval.")
        and event.activity_type.name != "agentkit.eval.plan"
        and event_id != capture_id
    ]
    assert len(evaluator_inputs) == 3
    for payload in [capture_payload, *evaluator_inputs]:
        if offloaded:
            [details] = payload.external_payloads
            assert details.size_bytes >= threshold
            claim = JSONProtoPayloadConverter().from_payload(payload, ExternalStorageReference)
            assert claim.driver_name == "archive"
        else:
            assert not payload.external_payloads
            assert payload.ByteSize() < threshold
            [events] = await client.data_converter.decode([payload], [list[AgentEvent]])
            assert AgentEvents(events).for_turn(first_turn.turn_id).reply_text == source_text
    [plan_id] = [
        event_id
        for event_id, event in scheduled.items()
        if event.activity_type.name == "agentkit.eval.plan"
    ]
    [plan_payload] = activity_results[plan_id].payloads
    assert not plan_payload.external_payloads
    for event_id, result in activity_results.items():
        if event_id not in (plan_id, capture_id):
            [score_payload] = result.payloads
            assert not score_payload.external_payloads
    [completion] = [
        event.workflow_execution_completed_event_attributes
        for event in history.events
        if event.HasField("workflow_execution_completed_event_attributes")
    ]
    [result_payload] = completion.result.payloads
    assert not result_payload.external_payloads
    if offloaded:
        assert all(source_text.encode() not in event.SerializeToString() for event in history.events)
