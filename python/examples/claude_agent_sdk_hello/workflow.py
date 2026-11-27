"""A streaming Claude Code conversation in an Agent Kit Workflow."""

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

    from temporal_agent_kit import agent
    from temporal_agent_kit.agent_protocol import (
        AgentConfig,
        MidTurn,
        TextMessage,
        TextReply,
        ToolApprovalPolicy,
    )
    from temporal_agent_kit.agent_workflow import (
        AgentWorkflowRunner,
        Injected,
        MessageContext,
    )
    from temporal_agent_kit.ai_sdks.claude_agent_sdk import (
        AgentKitClaudeSDKClient,
    )


TASK_QUEUE = "claude-agent-sdk-hello"


@agent.defn(name="ClaudeAgentSDKHello")
class ClaudeAgentSDKHelloWorkflow:
    @agent.init
    def __init__(self, config: AgentConfig) -> None:
        self._runner = AgentWorkflowRunner(
            config,
            stream=WorkflowStream(),
            approval_policy_default=ToolApprovalPolicy.allow_inherently_safe(),
        )
        self._client = AgentKitClaudeSDKClient(
            ClaudeAgentOptions(
                setting_sources=[],
                tools=["Agent", "Skill", "Write"],
            ),
            runner=self._runner,
        )

    @agent.setup
    async def connect(self) -> None:
        """Start Claude Code before the first message."""
        await self._client.connect()

    async def _ask(self, text: str) -> TextReply:
        """Run a turn for ``text``. The client repeats it if Claude Code's Worker is lost."""
        await self._client.query(text)
        async for message in self._client.receive_response():
            if isinstance(message, ResultMessage):
                if message.is_error:
                    raise ApplicationError(message.result or "")
                return TextReply(text=message.result or "")

        raise ApplicationError("Claude Code ended without a result.")

    @agent.accepts(mid_turn=MidTurn.ENQUEUE)
    async def ask(self, message: TextMessage) -> TextReply:
        """Ask Claude something. Sent while Claude is working, it waits for the current turn to end."""
        return await self._ask(message.text)

    @agent.accepts(mid_turn=MidTurn.ACCEPT)
    async def btw(self, message: TextMessage, ctx: Injected[MessageContext]) -> TextReply:
        """Steer Claude while it is working. Sent while Claude is idle, it is asked like `ask`."""
        if ctx.joined_turn:
            self._runner.steer(message.text)
            return TextReply(text="")
        return await self._ask(message.text)

    @agent.teardown
    async def disconnect(self) -> None:
        await self._client.disconnect()
