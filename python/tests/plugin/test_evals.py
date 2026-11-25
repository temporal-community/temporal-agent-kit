from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import httpx2
import openai
import pytest
from pydantic import ValidationError
from temporalio import workflow
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
    from examples.openai_hello import evals as hello_evals
    from examples.openai_hello.evals import count_failed_tools, total_input_tokens
    from temporal_agent_kit import AgentTurn, EvalSuiteResult, ScorerResult, Trajectory, agent
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
        TurnEnded,
        TurnStarted,
    )
    from temporal_agent_kit.ai_sdks.openai_agents import OpenAIAgentsPlugin
    from temporal_agent_kit.plugin import AgentKitPlugin
    from temporal_agent_kit.trajectory_activities import CapturedTurn
    from temporal_agent_kit.utils.large_payload import LocalFileStorageDriver
    from tests.conformance.script import (
        ModelStep,
        Script,
        SubagentSendStep,
        SubagentStartStep,
        ToolStep,
    )
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
        ({"status": "completed", "value": "1", "value_range": (0, 1)}, "value_range requires a numeric value"),
        ({"status": "skipped", "reasoning": "No reply", "value_range": (0, 1)}, "value_range requires a numeric value"),
        (
            {"status": "completed", "value": 0.5, "value_range": (1, 0)},
            "value_range lower bound must be less than or equal to its upper bound",
        ),
        ({"status": "completed", "value": -0.1, "value_range": (0, 1)}, "value must be within value_range"),
        ({"status": "completed", "value": 1.1, "value_range": (0, 1)}, "value must be within value_range"),
        ({"status": "completed", "value": 1, "error_kind": "reported"}, "error_kind is only allowed for error outcomes"),
    ],
)
def test_scorer_result_rejects_invalid_values_ranges_and_error_kinds(result, message):
    with pytest.raises(ValidationError, match=message):
        ScorerResult.model_validate(result)


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_scorer_result_accepts_numeric_values_within_an_inclusive_range(value):
    assert ScorerResult(status="completed", value=value, value_range=(0, 1)).value == value


def test_suite_definition_is_validated_at_import():
    async def undecorated(trajectory: Trajectory) -> ScorerResult:
        return ScorerResult(status="completed")

    with pytest.raises(TypeError, match="accepts only @agent.eval_scorer_defn"):

        @agent.eval_suite_defn
        class Undecorated:
            scorers = [undecorated]

    with pytest.raises(ValueError, match="duplicate scorer name"):

        @agent.eval_suite_defn
        class Duplicated:
            scorers = [count_failed_tools, count_failed_tools]

    with pytest.raises(TypeError, match="to list scorers"):

        @agent.eval_suite_defn
        class Empty:
            scorers: list[object] = []

    with pytest.raises(TypeError, match="provides it"):

        @agent.eval_suite_defn
        class AuthoredRun:
            scorers = [count_failed_tools]

            async def run(self, source: AgentTurn) -> None: ...

    with pytest.raises(TypeError, match=r"requires \(Trajectory\) -> ScorerResult"):

        @agent.eval_scorer_defn
        async def wrong_argument(events: list[AgentEvent]) -> ScorerResult:
            return ScorerResult(status="completed")


async def test_hello_judges_send_the_last_request_and_earlier_context_to_the_responses_api(monkeypatch):
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
        event(1, TurnStarted()),
        event(1, MessageHandlerEnd(output={"text": "Hello!"})),
        event(1, TurnEnded()),
    ]
    second = [
        event(2, message("What's the weather here?")),
        event(2, TurnStarted()),
        event(2, ToolStartEvent(
            tool_id="weather", tool_name="get_weather", tool_input={"city": "Seattle"}
        )),
        event(2, MessageHandlerEnd(output={"text": "It's 72°F and sunny in Seattle."})),
        event(2, TurnEnded()),
    ]

    def trajectory(turn_number: int, events: list[AgentEvent]) -> Trajectory:
        source = AgentTurn(workflow_id="hello", run_id="run", turn_number=turn_number)
        return Trajectory.from_capture(
            CapturedTurn(source=source, root_agent_id="hello", events=events)
        )

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
                            {"type": "output_text", "text": json.dumps(next(judgments)), "annotations": []}
                        ],
                    }
                ],
            },
        )

    client_class = openai.AsyncOpenAI

    def make_client() -> openai.AsyncOpenAI:
        return client_class(
            api_key="test",
            base_url="https://example.invalid/v1",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        )

    monkeypatch.setattr(openai, "AsyncOpenAI", make_client)

    first_turn = trajectory(1, first)
    later_turn = trajectory(2, first + second)
    assert later_turn.last_turn.all == second
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


async def _until_completed(agent: AgentClient, turn_number: int) -> None:
    async with asyncio.timeout(10):
        while True:
            status = await agent.get_status()
            if turn_number < status.current_turn or (
                turn_number == status.current_turn and not status.turn_active
            ):
                return
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("codec", [None, _WrappingCodec()])
@pytest.mark.parametrize(
    ("threshold", "offloaded"),
    [(3_000, True), (1_000_000, False)],
    ids=["above-threshold", "below-threshold"],
)
async def test_suite_uses_configured_storage_and_keeps_scorer_outcomes_independent(
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
        async with (
            Worker(
                client,
                task_queue="eval-source",
                workflows=[ScriptedAgent],
                workflow_runner=UnsandboxedWorkflowRunner(),
            ),
            Worker(client, task_queue="evals", workflows=[ScriptedAgentEvals]),
        ):
            handle = await client.start_workflow(
                ScriptedAgent.run, AgentConfig(), id="eval-agent", task_queue="eval-source"
            )
            assert handle.result_run_id is not None
            source = AgentClient(client, handle.id, run_id=handle.result_run_id)
            source_text = "A captured source reply. " * 200
            failing = Script(
                steps=[
                    ToolStep(id="failing-tool", name="fail", args={"message": "failed"}),
                    ModelStep(reply=[source_text], usage=TokenUsage(input_tokens=19)),
                ],
            )
            delegating = Script(
                steps=[
                    SubagentStartStep(name="child"),
                    SubagentSendStep(
                        name="child", script=Script(steps=[ModelStep(reply=["From the child."])])
                    ),
                ],
            )
            failed = await source.submit_message("run_enqueue", failing.model_dump(mode="json"))
            delegated = await source.submit_message(
                "run_enqueue", delegating.model_dump(mode="json")
            )
            await _until_completed(source, delegated.turn_number)
            first_turn = AgentTurn(
                workflow_id=handle.id,
                run_id=handle.result_run_id,
                turn_number=failed.turn_number,
            )
            second_turn = first_turn.model_copy(update={"turn_number": delegated.turn_number})
            failed_result: EvalSuiteResult = await client.execute_workflow(
                ScriptedAgentEvals.run, first_turn, id="eval-first", task_queue="evals"
            )
            second_eval = await client.start_workflow(
                ScriptedAgentEvals.run, second_turn, id="eval-second", task_queue="evals"
            )
            second_result = await second_eval.result()
            history = await second_eval.fetch_history()
        await Replayer(
            workflows=[ScriptedAgentEvals], data_converter=client.data_converter
        ).replay_workflow(history)

    assert failed_result.schema_version == 1
    assert failed_result.source == first_turn
    assert second_result.source == second_turn
    assert set(second_result.results) == {
        "clear_reply_text",
        "count_failed_tools",
        "crashing_inline_scorer",
        "invalid_inline_result",
        "prior_failed_tools",
        "reply_is_brief",
        "reported_activity_error",
        "reported_inline_error",
        "root_replying_agents",
        "total_input_tokens",
        "tree_replying_agents",
        "unavailable_judge",
    }
    assert failed_result.results["count_failed_tools"].value == 1
    assert second_result.results["count_failed_tools"].value == 0
    # clear_reply_text blanks every reply it is handed; each inline scorer scores its own copy.
    assert failed_result.results["clear_reply_text"].value == 1
    assert failed_result.results["reply_is_brief"].value == 0
    assert second_result.results["reply_is_brief"].status == "skipped"
    assert failed_result.results["total_input_tokens"].value == 19
    assert second_result.results["total_input_tokens"].value == 19
    assert failed_result.results["tree_replying_agents"].value == 1
    assert second_result.results["tree_replying_agents"].value == 2
    assert second_result.results["root_replying_agents"].value == 1
    invalid = second_result.results["invalid_inline_result"]
    assert invalid.status == "error"
    assert invalid.error_kind == "execution"
    assert invalid.value is None
    assert invalid.reasoning is not None and invalid.reasoning.startswith("InvalidScorerResult: ")
    assert "require reasoning and no score" in invalid.reasoning
    assert second_result.results["count_failed_tools"].error_kind is None
    previous = failed_result.results["prior_failed_tools"]
    assert previous == ScorerResult(status="skipped", reasoning="No earlier replies.")
    assert second_result.results["prior_failed_tools"].value == 1
    assert second_result.results["unavailable_judge"] == ScorerResult(
        status="error", error_kind="execution", reasoning="JudgeUnavailable: Judge unavailable."
    )
    assert second_result.results["crashing_inline_scorer"] == ScorerResult(
        status="error", error_kind="execution", reasoning="KeyError: 'text'"
    )
    assert second_result.results["reported_inline_error"] == ScorerResult(
        status="error", error_kind="reported", reasoning="Could not score inline."
    )
    assert second_result.results["reported_activity_error"] == ScorerResult(
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
    activity_results = {
        event.activity_task_completed_event_attributes.scheduled_event_id: (
            event.activity_task_completed_event_attributes.result
        )
        for event in history.events
        if event.HasField("activity_task_completed_event_attributes")
    }
    [capture_payload] = activity_results[capture_id].payloads
    scorer_inputs = [
        event.input.payloads[0]
        for event_id, event in scheduled.items()
        if event_id != capture_id
    ]
    assert len(scorer_inputs) == 4
    for payload in [capture_payload, *scorer_inputs]:
        if offloaded:
            [details] = payload.external_payloads
            assert details.size_bytes >= threshold
            claim = JSONProtoPayloadConverter().from_payload(payload, ExternalStorageReference)
            assert claim.driver_name == "archive"
        else:
            assert not payload.external_payloads
            assert payload.ByteSize() < threshold
            [capture] = await client.data_converter.decode([payload], [CapturedTurn])
            history_view = Trajectory.from_capture(capture).events
            assert history_view.for_turn(failed.turn_id).reply_text == source_text
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
    if offloaded:
        assert all(source_text.encode() not in event.SerializeToString() for event in history.events)
