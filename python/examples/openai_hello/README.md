# Hello-world OpenAI Agents agent

The smallest interesting agent you can build on Agent Kit with the **OpenAI Agents SDK**: it
chats in plain text and can call one tool (`get_weather`). It's here to show Agent Kit's
**streaming** path for the OpenAI integration end to end — from the model call, through the
streaming activity, to the live turn stream the web UI consumes.

## What it demonstrates

- **Streaming, not blocking.** The turn runs `Runner.run_streamed(...)` (not `Runner.run`).
  Only the streamed path routes model calls through `invoke_model_activity_streaming`, and it's
  that activity that feeds each raw OpenAI event to the Agent Kit observer.
- **The observer seam.** The worker builds the plugin with the two Agent Kit hooks:
  ```python
  OpenAIAgentsPlugin(
      model_params=ModelActivityParameters(stream_to_provider=stream_to_provider),
      observer_factory=kit_observer_factory,
  )
  ```
  `stream_to_provider` resolves the in-flight turn's stream context ambiently off the running
  workflow (no runner threading); `kit_observer_factory` turns it into the observer that
  translates raw OpenAI events into Agent Kit vocabulary — `model_interaction_started` →
  `reply_delta` … `tool_requested` … `model_interaction_ended`. There are **no run hooks**:
  the model-interaction bracket comes from the observer.
- **Agent Kit-owned tools.** `get_weather` is a normal `@agent.tool_defn`, adapted onto the SDK
  with `as_openai_agent_tool(...)`, so Agent Kit keeps approval + `tool_start`/`tool_end`.

## Evals authoring reference

[`evaluators.py`](evaluators.py) defines five evaluators that take `EvaluatorInput`.
The `OpenAIHelloEvals` suite lists the evaluators, and the worker registers the suite
alongside the agent Workflow.
The two judge Activities call the OpenAI Responses API directly for structured results.

| Evaluator | Execution | Result |
| --- | --- | --- |
| `count_failed_tools` | Activity | Number of failed tool calls on the selected turn |
| `full_context_understanding` | Activity with an OpenAI judge | Pass or fail on the use of relevant earlier context, or skipped when the turn is self-contained |
| `reply_is_brief` | Activity | Pass or fail against a 40-word limit per reply on the selected turn |
| `total_input_tokens` | Activity | Cumulative model input tokens through the selected turn, or skipped when usage is incomplete |
| `request_type` | Activity with an OpenAI judge | `greeting`, `weather_request`, `general_question`, or `other` for the selected request |

Try “Hi, I’m in Seattle” followed by “What’s the weather here?” Run evals on both
turns to see the request type change from greeting to weather request. The second
turn also gives the context judge a concrete reference to check.

After a turn completes, use a Kit-configured client and the agent worker's task queue
to run its evals:

```python
from examples.openai_hello.evaluators import OpenAIHelloEvals
from temporal_agent_kit import AgentTurn

result = await client.execute_workflow(
    OpenAIHelloEvals.run,
    AgentTurn(workflow_id=workflow_id, run_id=run_id, turn_number=turn_number),
    id=eval_workflow_id,
    task_queue="openai-hello",
)
print(result.results["count_failed_tools"])
print(result.results["request_type"])
```

Use the turn number returned when the request was accepted. See the
[evaluator guide](../../../docs/guides/agent-events.md) for how capture selects a run.

## Layout

| File | Role |
|---|---|
| `workflow.py` | `OpenAIHelloAgent` — the Agent Kit agent; one `ask` handler, one `get_weather` tool, driven by `Runner.run_streamed`. |
| `worker.py` | Worker hosting the workflow; wires the OpenAI plugin for the Agent Kit streaming seam, plus `AgentKitPlugin` for Agent Kit's own converter and activities. |
| `agents.toml`, `OpenAIHelloAgent.schema.json` | Registry entry that makes this agent selectable in the shared web UI, and the agent's schema it points at; generated from `BUILD.bazel` (`bazel run :agents_update`). |

There is **no per-example client**: like the Monty example, this agent is driven by the shared
example stack — the packaged FastAPI app and web UI (`temporal-agent-kit serve`), which send
messages and stream turns through Agent Kit's built-in `AgentClient`. Registering the agent (its `agent_kit_agent` in `BUILD.bazel`, which generates `agents.toml`) is all it takes to make it driveable.

## Run it

Prereq: from the repo root, `cp .env.example .env.local` and set `OPENAI_API_KEY` (and your
Temporal connection profile). Then, each in its own terminal:

```sh
just temporal          # 1. local Temporal dev server (or bring your own)
just server            # 2. builds the Svelte UI, then serves API + UI on http://localhost:8000
just worker            # 3. the agent worker
```

`just server` builds the web UI first, so it needs `pnpm` on your PATH. (If the build reports
missing modules, run `just app-install` once. For UI hot-reloading, `just ui-dev` runs it on Vite
with `/api` proxied to the server on :8000.)

Open http://localhost:8000, pick **OpenAI Hello**, and chat — e.g. *"What's the weather in
Paris?"*. The reply streams in token by token and the `get_weather` tool call appears live, each
event produced by the observer translating a raw OpenAI event in the streaming activity.

Without `just`, the equivalent commands (from the repo root):

```sh
uv run --group examples temporal-agent-kit serve examples/openai_hello/agents.toml --host 0.0.0.0 --port 8000
uv run --group examples python -m examples.openai_hello.worker
```
