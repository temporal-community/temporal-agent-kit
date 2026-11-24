# Read agent trajectories in evals

`AgentEvents` wraps native Temporal Agent Kit `AgentEvent` envelopes from one agent tree.
Pass it the events your evaluator retrieved, in stream order. The same wrapper works with
a root stream or a merged stream that includes subagents.

## Run evals for a turn

Declare an async function with the signature `(AgentEvents) -> EvaluatorResult` and
register it on `AgentKitPlugin(evaluators=[...])`. A list applies to every source agent
evaluated by that worker. Use a mapping keyed by source Workflow type to register
different evaluators on a shared worker.

- `@agent.eval_defn` runs the evaluator inline inside `RunEvaluatorsWorkflow`.
- `@agent.activity_eval_defn` runs the evaluator in its own Activity.

Start `RunEvaluatorsWorkflow.run` with an `AgentTurn` containing the source
`workflow_id`, `run_id`, and `turn_id`. The Kit selects the source agent's evaluators,
captures the retained events through that turn once, and uses that capture for both
kinds of evaluator. When External Storage is configured, the Temporal SDK applies
the configured size threshold and driver selection to the trajectory, plan, and
scores. Payloads below the threshold stay inline in Workflow history. For offloaded
payloads, history contains a native claim check and the SDK retrieves the data.
The Kit passes `AgentEvents` to the author function.

The capture ends at the requested turn's `TurnEnded` marker. Use
`events.for_turn(events.all[-1].turn_id)` to score that turn and `events.all` to inspect
earlier retained context. The capture reads the source agent's stream. Each subagent
has its own stream. An evaluator can return `skipped` or `error` with a reason.
An evaluator failure becomes an `error` result while other evaluator results are kept.
See the [OpenAI Hello evaluators](../../python/examples/openai_hello/evaluators.py).

## Inspect a retrieved trajectory

```python
from collections.abc import Iterable

from temporal_agent_kit import AgentEvents
from temporal_agent_kit.agent_protocol import AgentEvent, ReplyDelta, ToolErrorEvent


def inspect_reply(
    trajectory: Iterable[AgentEvent], turn_id: str, message_id: str
) -> None:
    events = AgentEvents(trajectory)
    turn = events.root_only().for_turn(turn_id)
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
- `root_only()` selects the root of the agent tree even if a subagent appears first.
  `for_agent(id)` selects an exact agent ID. `subagents()` returns a view per observed
  descendant. `subagents(agent_key=...)` uses the parent markers present in the view to
  match keys, so a partial trajectory may have child events with unknown keys.
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
- `model_call_count` counts observed starts. `total_usage` sums each field reported by
  every observed completion, across all agents in the current view. A missing count is
  `None`, and a reported zero stays zero. With no completions, a completion missing usage
  altogether, or a start that has yet to end in the view, all totals are unknown.
  `total_tokens` uses the provider's own totals.
- `thought_summaries` lists streamed thought-summary chunks. `steering_messages` lists
  enqueued steering text, including messages later dropped. `errors` reports handler,
  tool, and automatic approval evaluation errors. `close_reason` returns the reason for
  the sole observed close, or `None` if there is none. Select one agent when several closed.
