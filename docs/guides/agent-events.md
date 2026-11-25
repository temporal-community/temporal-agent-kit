# Evaluate agent sessions

An eval suite scores an agent session from its start through one completed turn. Its
scorers read that trajectory through `AgentEvents`, a read-only view over native `AgentEvent`
envelopes from one agent tree.

## Run an eval suite

Write each scorer as an async function with the signature `(Trajectory) -> ScorerResult`:

- `@agent.eval_scorer_defn` runs the scorer as Workflow code inside the eval suite's workflow.
  It follows Workflow rules: it must be deterministic, and it does any I/O, such as model
  calls, through Activities it starts. Use it for checks computed from the events alone, or for
  multi-step scorers where each step should be retried and recorded separately.
- `@agent.activity_eval_scorer_defn` runs the whole scorer in one Activity, retried as a unit.
  Use it to call a model or service directly.

Import `ScorerResult`, `Trajectory`, `AgentTurn` and `EvalSuiteResult` from `temporal_agent_kit`.
A completed result needs a value. A `value_range` is inclusive and needs a numeric value inside it.

Group scorers into an eval suite and register the suite on a worker like any workflow:

```python
from temporal_agent_kit import ScorerResult, Trajectory, agent


@agent.eval_scorer_defn
async def reply_is_brief(trajectory: Trajectory) -> ScorerResult:
    text = trajectory.last_turn.reply_text or ""
    return ScorerResult(status="completed", value=int(len(text.split()) <= 40))


@agent.eval_suite_defn(name="HelloEvals")
class HelloEvals:
    scorers = [reply_is_brief]
```

A suite does not name an agent. Start it with an `AgentTurn` holding the agent's
`workflow_id`, `run_id`, and `turn_number` to score any agent's session through that turn. With
`AgentKitPlugin` on the client, the worker registers the trajectory capture Activity and the
suite's Activity scorers. The result is an `EvalSuiteResult` with one `ScorerResult` per
scorer, keyed by function name.

```python
from temporal_agent_kit import AgentTurn

result = await client.execute_workflow(
    HelloEvals.run,
    AgentTurn(workflow_id=workflow_id, run_id=run_id, turn_number=turn_number),
    id=eval_workflow_id,
    task_queue=task_queue,
)
```

The suite checks the agent's status first and fails with `TurnNotCompleted` if the turn has
not completed. It then reads the session's events through that turn once, using
`AgentClient.attach` pinned to `run_id`, so a reused workflow ID never substitutes another run.
Every scorer receives the same capture. When External Storage is configured, the Temporal SDK
applies the configured size threshold and driver selection to the trajectory and scores.

A `Trajectory` holds:

- `events`: every captured event from the session's start through the end of the turn.
- `last_turn`: the same events narrowed to that final turn, a convenience for scorers
  that focus on it.
- `source` and `root_agent_id`: the turn the trajectory ends at and the agent that ran it.

A scorer judges the whole trajectory, so its score can reflect something that happened in
any earlier turn.

By default a scorer sees only the root agent's events. Pass `include_subagents=True` to the
scorer's decorator to score the whole agent tree: `events` and `last_turn` then also include
the events of every subagent the agent drove, in merged order. A subagent stream that cannot be
read fails the run instead of producing a partial tree.

A scorer can return `skipped` or `error` with a reason. The Kit sets `error_kind="reported"` when
a scorer returns an error. If a scorer raises, returns an invalid result, or cannot run, the Kit
records `error_kind="execution"` and includes the exception type in the reason. Activity failures
follow the Activity's retry policy before becoming results. The suite still returns the other
scorers' results.

An execution error is final for that eval run. After deploying a fix, run the suite again on the
same `AgentTurn`. See the
[OpenAI Hello scorers](../../python/examples/openai_hello/evals.py).

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
