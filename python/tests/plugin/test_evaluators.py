from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import httpx2
import openai
import pytest
from pydantic import ValidationError
from temporalio import activity, workflow
from temporalio.api.common.v1 import Payload
from temporalio.api.sdk.v1.external_storage_pb2 import ExternalStorageReference
from temporalio.converter import (
    DataConverter,
    ExternalStorage,
    JSONProtoPayloadConverter,
    PayloadCodec,
)
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker

with workflow.unsafe.imports_passed_through():
    from examples.openai_hello import evaluators as hello_evals
    from examples.openai_hello.evaluators import count_failed_tools, total_input_tokens
    from temporal_agent_kit import (
        AgentEvents,
        AgentTurn,
        EvalSuiteResult,
        EvaluatorInput,
        EvaluatorResult,
        agent,
    )
    from temporal_agent_kit.agent_client import AgentClient
    from temporal_agent_kit.agent_protocol import (
        AgentConfig,
        AgentEvent,
        AgentStreamItem,
        MessageAccepted,
        MessageDisposition,
        MessageHandlerEnd,
        TokenUsage,
        ToolStartEvent,
    )
    from temporal_agent_kit.ai_sdks.openai_agents import OpenAIAgentsPlugin
    from temporal_agent_kit.plugin import AgentKitPlugin
    from temporal_agent_kit.utils.large_payload import LocalFileStorageDriver
    from tests.conformance.script import ModelStep, Script, ToolStep
    from tests.conformance.scripted_agent import ScriptedAgent
    from tests.plugin.eval_suite_fixture import ScriptedAgentEvals


class _WrappingCodec(PayloadCodec):
    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload(metadata={"encoding": b"binary/test"}, data=item.SerializeToString()[::-1])
            for item in payloads
        ]

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        assert all(item.metadata == {"encoding": b"binary/test"} for item in payloads)
        return [Payload.FromString(item.data[::-1]) for item in payloads]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"status": "completed"}, "completed outcomes require a value"),
        (
            {"status": "completed", "value": "1", "value_range": (0, 1)},
            "value_range requires a numeric value",
        ),
        (
            {"status": "skipped", "reasoning": "No reply", "value_range": (0, 1)},
            "value_range requires a numeric value",
        ),
        (
            {"status": "completed", "value": 0.5, "value_range": (1, 0)},
            "value_range lower bound must be less than or equal to its upper bound",
        ),
        (
            {"status": "completed", "value": -0.1, "value_range": (0, 1)},
            "value must be within value_range",
        ),
        (
            {"status": "completed", "value": 1.1, "value_range": (0, 1)},
            "value must be within value_range",
        ),
    ],
)
def test_evaluator_result_rejects_invalid_values_and_ranges(result, message):
    with pytest.raises(ValidationError, match=message):
        EvaluatorResult.model_validate(result)


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_evaluator_result_accepts_numeric_values_within_inclusive_range(value):
    result = EvaluatorResult(status="completed", value=value, value_range=(0, 1))
    assert result.value == value


def test_suite_validates_evaluators_at_definition_time():
    async def undecorated(_):
        return 0

    with pytest.raises(TypeError, match="accepts only @agent.eval_defn"):

        @agent.eval_suite_defn
        class Undecorated:
            evaluators = (undecorated,)

    with pytest.raises(ValueError, match="duplicate evaluator name"):

        @agent.eval_suite_defn
        class Duplicated:
            evaluators = (count_failed_tools, count_failed_tools)

    with pytest.raises(TypeError, match=r"requires a non-empty Empty\.evaluators"):

        @agent.eval_suite_defn
        class Empty:
            evaluators = ()

    with pytest.raises(TypeError, match="defines run"):

        @agent.eval_suite_defn
        class AuthoredRun:
            evaluators = (count_failed_tools,)

            async def run(self, source: AgentTurn) -> None: ...


async def test_hello_judges_receive_the_selected_request_and_earlier_context(monkeypatch):
    def event(turn: int, payload: AgentStreamItem) -> AgentEvent:
        return AgentEvent(
            agent_id="hello",
            turn_id=f"turn-{turn}",
            turn_number=turn,
            message_id=f"msg-{turn}",
            timestamp=turn,
            event=payload,
        )

    def message(text: str) -> MessageAccepted:
        return MessageAccepted(
            handler="ask", payload={"text": text}, disposition=MessageDisposition.OPENED
        )

    first = [
        event(1, message("Hi, I'm in Seattle.")),
        event(1, MessageHandlerEnd(output={"text": "Hello!"})),
    ]
    second = [
        event(2, message("What's the weather here?")),
        event(2, ToolStartEvent(
            tool_id="weather", tool_name="get_weather", tool_input={"city": "Seattle"}
        )),
        event(2, MessageHandlerEnd(output={"text": "It's 72°F and sunny in Seattle."})),
    ]
    judged = []
    formats = []
    judgments = iter(
        [
            {"request_type": "greeting", "reason": "Introduced their location."},
            {"verdict": "pass", "reason": "The lookup used Seattle."},
            {"request_type": "weather_request", "reason": "Asked for local weather."},
        ]
    )

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        judged.append(json.loads(body["input"]))
        response_format = body["text"]["format"]
        assert response_format["type"] == "json_schema"
        formats.append(response_format["name"])
        return httpx2.Response(
            200,
            json={
                "id": "resp_test",
                "created_at": 0,
                "model": body["model"],
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(next(judgments)),
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    def make_client() -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key="test",
            base_url="https://example.invalid/v1",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        )

    monkeypatch.setattr(hello_evals, "AsyncOpenAI", make_client)

    trailing = event(3, ToolStartEvent(tool_id="later", tool_name="get_weather", tool_input={}))
    child_message = event(2, message("Internal subagent request.")).model_copy(
        update={"agent_id": "hello-child", "turn_id": "child-turn"}
    )
    first_turn = EvaluatorInput(turn_id="turn-1", events=AgentEvents(first))
    later_turn = EvaluatorInput(
        turn_id="turn-2", events=AgentEvents([*first, *second, child_message, trailing])
    )
    assert later_turn.turn_events.all == second
    assert (await hello_evals.full_context_understanding(first_turn)).status == "skipped"
    assert (await hello_evals.request_type(first_turn)).value == "greeting"
    assert (await hello_evals.full_context_understanding(later_turn)).value == 1
    assert (await hello_evals.request_type(later_turn)).value == "weather_request"
    assert (await total_input_tokens(later_turn)).status == "skipped"
    assert formats == ["_RequestClassification", "_ContextJudgment", "_RequestClassification"]
    earlier = [
        {"role": "user", "text": "Hi, I'm in Seattle."},
        {"role": "assistant", "text": "Hello!"},
    ]
    assert judged == [
        {"earlier": [], "requests": ["Hi, I'm in Seattle."]},
        {
            "earlier": earlier,
            "requests": ["What's the weather here?"],
            "replies": ["It's 72°F and sunny in Seattle."],
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "Seattle"}}],
        },
        {"earlier": earlier, "requests": ["What's the weather here?"]},
    ]


@pytest.mark.parametrize(
    ("threshold", "codec", "offloaded"),
    [(3_000, _WrappingCodec(), True), (1_000_000, None, False)],
    ids=["offloaded-with-codec", "inline"],
)
async def test_suite_uses_configured_storage_and_keeps_evaluator_outcomes_independent(
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
    async with await WorkflowEnvironment.start_local(
        plugins=[OpenAIAgentsPlugin(register_activities=False), AgentKitPlugin()],
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
            Worker(
                client,
                task_queue="evals",
                workflows=[ScriptedAgentEvals, hello_evals.OpenAIHelloEvals],
                activities=[application_capture],
            ),
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
                    ModelStep(reply=[source_text], usage=TokenUsage(input_tokens=19)),
                ],
            )
            failed = await source.submit_message("run_enqueue", script.model_dump(mode="json"))
            clean = await source.submit_message("run_enqueue", {"steps": []})
            async with asyncio.timeout(5):
                while True:
                    status = await source.get_status()
                    if status.current_turn == clean.turn_number and not status.turn_active:
                        break
                    await asyncio.sleep(0.01)
            first_turn = AgentTurn(
                workflow_id=handle.id, run_id=handle.result_run_id, turn_number=failed.turn_number
            )
            failed_eval = await client.start_workflow(
                ScriptedAgentEvals.run, first_turn, id="eval-first", task_queue="evals"
            )
            clean_eval = await client.start_workflow(
                ScriptedAgentEvals.run,
                first_turn.model_copy(update={"turn_number": clean.turn_number}),
                id="eval-second",
                task_queue="evals",
            )
            failed_result = await failed_eval.result()
            clean_result = await clean_eval.result()
            history = await clean_eval.fetch_history()
        await Replayer(
            workflows=[ScriptedAgentEvals], data_converter=client.data_converter
        ).replay_workflow(history)

    assert isinstance(failed_result, EvalSuiteResult)
    assert failed_result.schema_version == 1
    assert failed_result.source == first_turn
    assert clean_result.source.turn_number == clean.turn_number
    assert set(clean_result.results) == {
        "clear_reply_text",
        "count_failed_tools",
        "crashing_inline_evaluator",
        "inline_reply_length",
        "invalid_inline_result",
        "prior_failed_tools",
        "reply_is_brief",
        "reported_activity_error",
        "reported_inline_error",
        "total_input_tokens",
        "unavailable_judge",
    }
    assert failed_result.results["clear_reply_text"].value == 1
    assert failed_result.results["inline_reply_length"].value == len(source_text)
    assert failed_result.results["count_failed_tools"].value == 1
    assert clean_result.results["count_failed_tools"].value == 0
    assert clean_result.results["count_failed_tools"].error_kind is None
    assert failed_result.results["reply_is_brief"].value == 0
    assert clean_result.results["reply_is_brief"].status == "skipped"
    assert failed_result.results["total_input_tokens"].value == 19
    assert clean_result.results["total_input_tokens"].value == 19
    invalid = clean_result.results["invalid_inline_result"]
    assert invalid.status == "error"
    assert invalid.value is None
    assert invalid.error_kind == "execution"
    assert invalid.reasoning == (
        "InvalidEvaluatorResult: Value error, skipped and error outcomes require reasoning and no score"
    )
    previous = failed_result.results["prior_failed_tools"]
    assert previous.status == "skipped"
    assert previous.reasoning == "No earlier replies."
    assert previous.value is None
    assert previous.error_kind is None
    assert clean_result.results["prior_failed_tools"].value == 1
    assert clean_result.results["unavailable_judge"] == EvaluatorResult(
        status="error", error_kind="execution", reasoning="JudgeUnavailable: Judge unavailable."
    )
    assert clean_result.results["crashing_inline_evaluator"] == EvaluatorResult(
        status="error", error_kind="execution", reasoning="KeyError: 'text'"
    )
    assert clean_result.results["reported_inline_error"] == EvaluatorResult(
        status="error", error_kind="reported", reasoning="Could not score inline."
    )
    assert clean_result.results["reported_activity_error"] == EvaluatorResult(
        status="error", error_kind="reported", reasoning="Could not score in the Activity."
    )

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
        event.input.payloads
        for event_id, event in scheduled.items()
        if event_id != capture_id
    ]
    assert len(evaluator_inputs) == 6
    trajectory_payloads = [capture_payload]
    for events_payload, turn_id_payload in evaluator_inputs:
        assert await client.data_converter.decode([turn_id_payload], [str]) == [clean.turn_id]
        trajectory_payloads.append(events_payload)
    for payload in trajectory_payloads:
        if offloaded:
            [details] = payload.external_payloads
            assert details.size_bytes >= threshold
            claim = JSONProtoPayloadConverter().from_payload(payload, ExternalStorageReference)
            assert claim.driver_name == "archive"
        else:
            assert not payload.external_payloads
            assert payload.ByteSize() < threshold
        [events] = await client.data_converter.decode([payload], [list[AgentEvent]])
        assert AgentEvents(events).for_turn(failed.turn_id).reply_text == source_text
    for event_id, result in activity_results.items():
        if event_id != capture_id:
            [score_payload] = result.payloads
            assert not score_payload.external_payloads
    [completion] = [
        event.workflow_execution_completed_event_attributes
        for event in history.events
        if event.HasField("workflow_execution_completed_event_attributes")
    ]
    [result_payload] = completion.result.payloads
    assert not result_payload.external_payloads
