# Claude Agent SDK Temporal Integration


Install with `uv add 'temporal-agent-kit[claude-agent-sdk]'`. Import `ClaudeAgentSDKPlugin` and
`TemporalTransport` from `temporal_agent_kit.ai_sdks.claude_agent_sdk` when following the
[upstream usage guide](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk).

## Agent Kit

Register `ClaudeAgentSDKPlugin(event_publisher=AgentKitEventPublisher)` and `AgentKitPlugin()` on
the Worker, and use `AgentKitClaudeSDKClient` as you would `ClaudeSDKClient`. It streams Claude's replies,
thinking, and model calls to the Agent Kit turn. Create it in `@agent.init`, connect it in `@agent.setup`, and disconnect it in
`@agent.teardown`:

```python
@agent.init
def __init__(self, config: AgentConfig) -> None:
    self._runner = AgentWorkflowRunner(config, stream=WorkflowStream())
    self._client = AgentKitClaudeSDKClient(ClaudeAgentOptions(), runner=self._runner)

@agent.setup
async def connect(self) -> None:
    await self._client.connect()

@agent.accepts()
async def ask(self, message: TextMessage) -> TextReply:
    await self._client.query(message.text)
    async for response in self._client.receive_response():
        if isinstance(response, ResultMessage):
            if response.is_error:
                raise ApplicationError(response.result or "")
            return TextReply(text=response.result or "")
    raise ApplicationError("Claude Code ended without a result.")

@agent.teardown
async def disconnect(self) -> None:
    await self._client.disconnect()
```

If the Worker running Claude Code is lost, or Claude Code stops mid-turn, the client repeats the
turn: it starts Claude Code again after the last finished turn, and sends the prompt again.

See the [Claude Agent SDK Hello example](../../../examples/claude_agent_sdk_hello).

## Streaming

To stream to Agent Kit, pass `ClaudeAgentSDKPlugin(event_publisher=AgentKitEventPublisher)` and
register `AgentKitPlugin()` on the same Worker.

```python
worker = Worker(
    client,
    task_queue="my-agent",
    workflows=[MyAgentWorkflow],
    plugins=[
        ClaudeAgentSDKPlugin(event_publisher=AgentKitEventPublisher),
        AgentKitPlugin(),
    ],
)
```

## Activities on the private Worker Entity

To run Activities on the private Worker Entity that runs Claude Code, such as downloading files
from S3 before Claude Code starts, register them on the plugin:

```python
ClaudeAgentSDKPlugin(hook_activities=[stage_files], event_publisher=AgentKitEventPublisher)
```

Then call them from a transport hook, which receives the reserved private Task Queue:

```python
async def stage(task_queue: str) -> None:
    await workflow.execute_activity(
        stage_files,
        task_queue=task_queue,
        start_to_close_timeout=timedelta(minutes=5),
    )

client = AgentKitClaudeSDKClient(runner=self._runner, hooks={"before_connect": stage})
```

To run something after each successful turn, such as uploading files to S3, pass an `after_turn`
hook. It also receives the reserved private Task Queue, and runs once for a turn that finished, not for one
that was lost and repeated:

```python
client = AgentKitClaudeSDKClient(runner=self._runner, hooks={"after_turn": upload})
```
