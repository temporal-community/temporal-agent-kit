# Claude Agent SDK Temporal Integration


Install with `uv add 'temporal-agent-kit[claude-agent-sdk]'`. Import `ClaudeAgentSDKPlugin` and
`TemporalTransport` from `temporal_agent_kit.ai_sdks.claude_agent_sdk` when following the
[upstream usage guide](https://docs.temporal.io/develop/python/integrations/claude-agent-sdk).

## Activities on the private Worker Entity

To run Activities on the private Worker Entity that runs Claude Code, such as downloading files
from S3 before Claude Code starts, register them on the plugin:

```python
ClaudeAgentSDKPlugin(hook_activities=[stage_files])
```

Then call them from a transport hook, which receives the reserved private Task Queue:

```python
async def stage(task_queue: str) -> None:
    await workflow.execute_activity(
        stage_files,
        task_queue=task_queue,
        start_to_close_timeout=timedelta(minutes=5),
    )

client = TemporalClaudeSDKClient(hooks={"before_connect": stage})
```

To run something after each successful turn, such as uploading files to S3, pass an `after_turn`
hook. It also receives the reserved private Task Queue, and runs once for a turn that finished, not for one
that was lost and repeated:

```python
client = AgentKitClaudeSDKClient(runner=self._runner, hooks={"after_turn": upload})
```
