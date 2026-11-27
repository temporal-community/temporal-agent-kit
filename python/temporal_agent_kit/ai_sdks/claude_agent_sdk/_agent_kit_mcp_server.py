"""Serve Agent Kit tools to Claude Code."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from claude_agent_sdk import McpSdkServerConfig
from mcp import types
from mcp.server import Server
from pydantic import TypeAdapter
from pydantic_core import to_json

from temporal_agent_kit.agent_workflow import AgentWorkflowRunner

_HARNESS_TOOL_ATTRS = ("__agent_tool__", "__agent_activity_tool__")

AGENT_KIT_MCP_SERVER_NAME = "agent_kit"
"""The ``mcp_servers`` key the Agent Kit server is registered under. Claude Code prefixes its tools with ``mcp__agent_kit__``."""

_CLAUDE_CODE_TOOL_PREFIX = f"mcp__{AGENT_KIT_MCP_SERVER_NAME}__"


class AgentKitMcpServer(Server):
    """An MCP server that runs Agent Kit tools through a runner. The tools publish their own Agent Kit events."""

    def __init__(
        self,
        runner: AgentWorkflowRunner,
        tools: Sequence[Callable[..., Awaitable[Any]]],
        *,
        injections: Mapping[str, Any] | None = None,
    ) -> None:
        for tool in tools:
            if not any(getattr(tool, attr, False) for attr in _HARNESS_TOOL_ATTRS):
                raise TypeError(f"{tool!r} is not an Agent Kit tool.")

        self._runner = runner
        self._tools = {tool.__name__: tool for tool in tools}
        self._injections = injections
        super().__init__(AGENT_KIT_MCP_SERVER_NAME, on_list_tools=self._list_tools, on_call_tool=self._call_tool)

    async def _list_tools(self, ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=tool_name,
                    description=inspect.getdoc(tool) or "",
                    # No Injected parameters: Agent Kit's tool decorators already removed them from the signature.
                    input_schema=TypeAdapter(tool).json_schema(),
                )
                for tool_name, tool in self._tools.items()
            ]
        )

    async def _call_tool(self, ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            tool = self._tools.get(params.name)
            if tool is None:
                raise ValueError(f"Tool {params.name!r} not found.")

            tool_use_id = (params.meta or {}).get("claudecode/toolUseId")
            if tool_use_id is None:
                raise ValueError("Claude Code sent no tool use ID.")

            result = await self._runner.run_tool(
                tool_use_id, tool, injections=self._injections, **(params.arguments or {})
            )
            text = result if isinstance(result, str) else to_json(result, fallback=str).decode()
            return types.CallToolResult(content=[types.TextContent(type="text", text=text)])
        except Exception as error:  # noqa: BLE001 - the model sees a failed call as an error result
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(error))], is_error=True)


def agent_kit_mcp_server(
    runner: AgentWorkflowRunner,
    tools: Sequence[Callable[..., Awaitable[Any]]],
    *,
    injections: Mapping[str, Any] | None = None,
) -> McpSdkServerConfig:
    """An in-process MCP server that runs Agent Kit tools through ``runner``.

    Each call runs under Claude Code's tool use ID, so its Agent Kit events match the call's ToolRequested.
    The runner supplies ``injections`` to the tools' ``Injected`` parameters by name.
    """
    server = AgentKitMcpServer(runner, tools, injections=injections)
    return McpSdkServerConfig(type="sdk", name=AGENT_KIT_MCP_SERVER_NAME, instance=server)


def agent_kit_tool_name(claude_code_name: str) -> str | None:
    """The Agent Kit name of a Claude Code tool, or ``None`` if the tool is not on the Agent Kit server."""
    if not claude_code_name.startswith(_CLAUDE_CODE_TOOL_PREFIX):
        return None
    return claude_code_name.removeprefix(_CLAUDE_CODE_TOOL_PREFIX)
