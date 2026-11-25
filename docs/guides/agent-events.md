# Read agent trajectories in evals

`AgentEvents` wraps native Temporal Agent Kit `AgentEvent` envelopes from one agent tree.
Pass it the events your evaluator retrieved, in stream order. The same wrapper works with
a root stream or a merged stream that includes subagents.

## Run an eval suite for a turn

Import `EvaluatorInput` and `EvaluatorResult` from `temporal_agent_kit`. Declare an async
evaluator with the signature `(EvaluatorInput) -> EvaluatorResult`.

- `@agent.eval_defn` runs the evaluator inline as Workflow code inside the eval suite.
  It follows Workflow rules: it must be deterministic, and it does any I/O, such as model
  calls, through Activities it starts. Use it for checks computed from the events alone, or
  for multi-step evals where each step should be retried and recorded separately.
- `@agent.activity_eval_defn` runs the whole evaluator in one Activity, retried as a unit.
  Use it to call a model or service directly.

Group evaluators into a suite with `@agent.eval_suite_defn`. Register the suite in the
worker's `workflows` list with `AgentKitPlugin()` on the client. The plugin registers
the capture Activity and the suite's Activity-backed evaluators.

```python
@agent.eval_suite_defn
class HelloEvals:
    evaluators = (reply_is_brief, request_type)
```

The caller selects the suite by starting `HelloEvals.run`. A worker can host several suites,
and a suite can evaluate turns from several agent types. Pass an `AgentTurn` containing
the source `workflow_id`, `run_id`, and `turn_number`. Capture reads the specified run
even if the Workflow ID has been reused. If the turn hasn't completed, capture fails with
`TurnNotCompleted` before reading the stream. The Kit captures the retained events through that turn
once, runs the suite's evaluators, and returns an `EvalSuiteResult` with the outcome
of each evaluator keyed by name. When External Storage is configured, the Temporal SDK applies
the configured size threshold and driver selection to the trajectory and
scores. Payloads below the threshold stay inline in Workflow history. For offloaded
payloads, history contains a native claim check and the SDK retrieves the data.
The capture ends at the requested turn's `TurnEnded` marker. The evaluator input carries the
requested `turn_id`. Use `evaluation.turn_events` to score that turn and `evaluation.events`
to inspect the full capture, including earlier retained context. Both are `AgentEvents` views.
Capture merges retained subagent turns into `evaluation.events`. The merger uses
dispatch and reply events to keep child turns in order, including nested subagents.
If a retained dispatch or reply points to a child turn whose start or end is missing,
capture fails with `IncompleteTrajectory`.

Use `evaluation.turn_events` for the requested turn, including the subagent events merged
between its start and end. Use `evaluation.turn_events.parent_events` for the source agent's
events in that turn, or `evaluation.events.parent_events` across the capture. Subagent events
keep their own agent and turn IDs and can be selected with `evaluation.events.for_agent(subagent_id)`.
See the [OpenAI Hello evaluators](../../python/examples/openai_hello/evaluators.py).

If your agent uses Continue-As-New, carry its `WorkflowStreamState` and turn numbering into
the successor run to evaluate earlier turns using that run's ID.

An evaluator can return `skipped` or `error` with a reason. The Kit sets
`error_kind="reported"` when an evaluator returns an error. If an evaluator raises,
returns an invalid result, or cannot load, the Kit records `error_kind="execution"`
and includes the exception type in the reason. Activity failures follow the Activity's
retry policy before becoming results. The runner still returns the other evaluators' results.

An execution error is final for that eval run. After deploying a fix, start the suite
again for the same `AgentTurn` while the source stream still retains it.

## Inspect a retrieved trajectory

```python
from collections.abc import Iterable

from temporal_agent_kit import AgentEvents
from temporal_agent_kit.agent_protocol import AgentEvent, ReplyDelta, ToolErrorEvent


def inspect_reply(
    trajectory: Iterable[AgentEvent], agent_id: str, turn_id: str, message_id: str
) -> None:
    events = AgentEvents(trajectory)
    turn = events.for_agent(agent_id).for_turn(turn_id)
    message = turn.for_message(message_id)

    print(message.reply_text)
    for item in message.of_type(ReplyDelta).all:
        if isinstance(item.event, ReplyDelta):
            print(item.event.text)
    print(turn.total_usage.input_tokens)

    print(len(message.of_type(ToolErrorEvent).all))
```

## Choose the events to inspect

- `all` gives you the original envelopes with their agent, turn, message, and timestamp.
  `of_type(...)` returns an `AgentEvents` view and chains with other filters. Use
  `.all` on the view to read the matching envelopes in stream order and `.event`
  on an envelope to read its payload. Treat the shared envelopes and payloads as read-only.
- `for_agent(id)` selects events emitted by an exact agent ID.
- `for_turn(id)` and `for_message(id)` match envelope IDs exactly. A child dispatched during
  a root turn has its own turn ID. To inspect all agents in a merged trajectory, use the
  original view. A single turn can contain several inbound messages.
- `replies` lists every successful handler completion with its envelope, including
  structured outputs. `reply_text` reads the string `text` field from the sole completion.
  It returns `None` if no such field or completion is present and raises `ValueError` if
  several replies completed. Use `ReplyDelta` payloads to inspect streamed text separately.

## Inspect tools, usage, and diagnostics

- Use `of_type(ToolErrorEvent)` for tool failures or `of_type(ToolEndEvent)` for
  completed tool calls and their outputs.
- `model_call_count` counts observed model starts. `total_usage` sums each field reported by
  every observed completion, across all agents in the current view. A missing count is
  `None`, and a reported zero stays zero. With no completions, a completion missing usage
  altogether, or a start that has yet to end in the view, all totals are unknown.
  `total_tokens` uses the provider's own totals.
- `errors` lists handler, tool, and automatic approval evaluation error messages in stream
  order. `close_reason` returns the sole observed close reason, or `None` if there is none.
  Select one agent when several closed.
