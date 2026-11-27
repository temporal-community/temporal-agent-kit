"""A ``ClaudeSDKClient`` wired to Agent Kit streaming and interrupt recovery."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from claude_agent_sdk.types import (
    HookCallback,
    HookContext,
    HookInput,
    HookJSONOutput,
    McpServerConfig,
)
from temporal_agent_kit.stream_context import TurnStreamContext
from temporalio.exceptions import ApplicationError

from temporal_agent_kit.agent_protocol import (
    AgentEventType,
    AgentStreamItem,
    ModelInteractionEnded,
    ToolEndEvent,
    ToolErrorEvent,
    ToolStartEvent,
)
from temporal_agent_kit.agent_workflow import (
    _CURRENT_STREAM_CONTEXT,
    _CURRENT_MESSAGE_ID,
    AgentWorkflowRunner,
    ToolApprovalDenied,
    _apply_approval_policy,
)

from temporal_agent_kit.ai_sdks.claude_agent_sdk._agent_kit_mcp_server import (
    AGENT_KIT_MCP_SERVER_NAME,
    agent_kit_mcp_server,
    agent_kit_tool_name,
)
from temporal_agent_kit.ai_sdks.claude_agent_sdk._client import TemporalClaudeSDKClient
from temporal_agent_kit.ai_sdks.claude_agent_sdk._types import TransportHooks



class AgentKitClaudeSDKClient(TemporalClaudeSDKClient):
    """A :class:`TemporalClaudeSDKClient` wired to stream turns and handle interrupts with Agent Kit."""

    def __init__(
        self,
        options: ClaudeAgentOptions | None = None,
        *,
        runner: AgentWorkflowRunner,
        tools: Sequence[Callable[..., Awaitable[Any]]] = (),
        injections: Mapping[str, Any] | None = None,
        **transport_kwargs: Any,
    ) -> None:
        """Prepare the client. Claude Code starts on ``connect()``.

        Args:
            options: Claude Agent SDK options. Defaults to ``ClaudeAgentOptions()``.
            runner: The Agent Kit runner to stream turns to.
            tools: Agent Kit tools to serve to Claude Code.
            **transport_kwargs: ``hooks`` and ``activity_options`` for the :class:`TemporalTransport`.
        """
        options = options or ClaudeAgentOptions()
        if tools and not isinstance(options.mcp_servers, dict):
            raise ApplicationError("mcp_servers is only supported as a dict, not a config file path.")

        if isinstance(options.mcp_servers, dict) and AGENT_KIT_MCP_SERVER_NAME in options.mcp_servers:
            raise ApplicationError(
                f"The MCP server name {AGENT_KIT_MCP_SERVER_NAME!r} is reserved for Agent Kit tools."
            )

        self._runner = runner

        # Built-in tools Claude Code started and that have not finished, by tool use ID. The runner tracks Agent Kit tools.
        self._running_tools: dict[str, str] = {}

        self._context: TurnStreamContext | None = None

        super().__init__(
            self._options(options, tools, injections),
            hooks=self._transport_hooks(transport_kwargs.pop("hooks", None)),
            **transport_kwargs,
        )

    async def disconnect(self) -> None:
        """Fail the built-in tools still running, end the model call, and close the connection."""
        if self._query is not None and self._runner.current_stream_context is not None:
            for tool_id, tool_name in self._running_tools.items():
                self._publish(
                    ToolErrorEvent(tool_id=tool_id, tool_name=tool_name, message="Claude Code connection closed.")
                )
            self._publish(ModelInteractionEnded())
        self._running_tools.clear()
        await super().disconnect()

    def _options(
        self,
        base: ClaudeAgentOptions,
        tools: Sequence[Callable[..., Awaitable[Any]]],
        injections: Mapping[str, Any] | None,
    ) -> ClaudeAgentOptions:
        post_tool_hooks = HookMatcher(hooks=[self._steering_claude_hook(self._runner), self._builtin_tools_end_claude_hook])
        hooks = dict(base.hooks or {})
        hooks["PreToolUse"] = [*hooks.get("PreToolUse", []), HookMatcher(hooks=[self._builtin_tools_start_claude_hook])]
        for event in ("PostToolUse", "PostToolUseFailure"):
            hooks[event] = [*hooks.get(event, []), post_tool_hooks]

        overrides: dict[str, Any] = {
            "include_partial_messages": True,
            "hooks": hooks,
            "mcp_servers": self._mcp_servers(base.mcp_servers, tools, injections),
        }
        if base.thinking is None and base.max_thinking_tokens is None:
            overrides["thinking"] = {"type": "adaptive", "display": "summarized"}
        if base.can_use_tool is None and base.permission_prompt_tool_name is None:
            overrides["can_use_tool"] = self._can_use_tool
            if base.permission_mode is None:
                overrides["permission_mode"] = "default"
        return dataclasses.replace(base, **overrides)

    def _mcp_servers(
        self,
        base: dict[str, McpServerConfig] | str | Path,
        tools: Sequence[Callable[..., Awaitable[Any]]],
        injections: Mapping[str, Any] | None,
    ) -> dict[str, McpServerConfig] | str | Path:
        """``base`` with the Agent Kit server added, if there are tools. ``__init__`` checked that it can be."""
        if not tools:
            return base
        assert isinstance(base, dict)
        return {**base, AGENT_KIT_MCP_SERVER_NAME: agent_kit_mcp_server(self._runner, tools, injections=injections)}

    async def _can_use_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Gate built-in tools on the approval policy. Stop tracking the ones that will not run."""
        tool_id = context.tool_use_id
        assert tool_id is not None

        if agent_kit_tool_name(tool_name) is not None:  # Agent Kit tools gate themselves in the runner
            return PermissionResultAllow()

        try:
            await self._runner.run_tool(tool_id, _apply_approval_policy, tool_name, tool_input, inherently_safe=False)
        except ToolApprovalDenied as error:
            self._running_tools.pop(tool_id, None)
            return PermissionResultDeny(message=str(error))
        except asyncio.CancelledError:  # Claude Code withdrew the request, or the connection is closing
            self._running_tools.pop(tool_id, None)
            raise

        self._publish(ToolStartEvent(tool_id=tool_id, tool_name=tool_name, tool_input=tool_input))
        return PermissionResultAllow()

    async def _builtin_tools_start_claude_hook(
        self,
        hook_input: HookInput,
        tool_use_id: str | None,
        _context: HookContext,
    ) -> HookJSONOutput:
        """Track each built-in tool Claude Code is about to run until it finishes.

        Runs also for tools Claude Code runs without asking can_use_tool.
        """
        assert tool_use_id is not None and hook_input["hook_event_name"] == "PreToolUse"

        tool_name = hook_input["tool_name"]
        if agent_kit_tool_name(tool_name) is None:
            self._running_tools[tool_use_id] = tool_name
        return {}

    async def _builtin_tools_end_claude_hook(
        self,
        hook_input: HookInput,
        tool_use_id: str | None,
        _context: HookContext,
    ) -> HookJSONOutput:
        assert tool_use_id is not None

        tool_name = self._running_tools.pop(tool_use_id, None)
        if tool_name is None:  # Agent Kit tools, which the runner tracks
            return {}

        if hook_input["hook_event_name"] == "PostToolUse":
            self._publish(
                ToolEndEvent(tool_id=tool_use_id, tool_name=tool_name, tool_output=hook_input["tool_response"])
            )
        elif hook_input["hook_event_name"] == "PostToolUseFailure":
            self._publish(ToolErrorEvent(tool_id=tool_use_id, tool_name=tool_name, message=hook_input["error"]))
            
        return {}

    def _steering_claude_hook(self, runner: AgentWorkflowRunner) -> HookCallback:
        async def claim_steering(
            hook_input: HookInput,
            tool_use_id: str | None,
            _context: HookContext,
        ) -> HookJSONOutput:              
            steering = runner.claim_steering(tool_id=tool_use_id)
            if steering is None:
                return {}

            if hook_input["hook_event_name"] == "PostToolUse":
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": steering.render(),
                    }
                }

            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUseFailure",
                    "additionalContext": steering.render(),
                }
            }

        return claim_steering

    def _transport_hooks(self, base: TransportHooks | None) -> TransportHooks:
        hooks: TransportHooks = {**(base or {})}

        original_before_write = hooks.get("before_write")
        async def before_write(task_queue: str) -> None:
            context = _CURRENT_STREAM_CONTEXT.get()
            if context is not None:
                self._context = context

            if original_before_write is not None:
                await original_before_write(task_queue)
        hooks["before_write"] = before_write

        original_on_output = hooks.get("on_output")
        async def on_output(task_queue: str) -> None:
            if self._context is not None:
                _CURRENT_STREAM_CONTEXT.set(self._context)
                _CURRENT_MESSAGE_ID.set(self._context.message_id)

            if original_on_output is not None:
                await original_on_output(task_queue)
        hooks["on_output"] = on_output

        return hooks

    def _publish(self, event: AgentStreamItem):
        if self._context:
            self._runner._pub(
                self._context.turn_id, self._context.turn_number, event, message_id=self._context.message_id
            )
