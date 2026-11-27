"""Carry the running turn from the Workflow to its Activities in a header."""

from __future__ import annotations

from typing import Any

from temporalio import workflow
from temporalio.converter import PayloadConverter
from temporalio.worker import (
    ActivityInboundInterceptor,
    ExecuteActivityInput,
    Interceptor,
    StartActivityInput,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    WorkflowOutboundInterceptor,
)

from temporal_agent_kit.agent_workflow import _CURRENT_STREAM_CONTEXT
from temporal_agent_kit.stream_context import TurnStreamContext

_HEADER = "temporal-agent-kit-stream-context"


class StreamContextInterceptor(Interceptor):
    """Send ``_CURRENT_STREAM_CONTEXT`` with each Activity a Workflow starts, and set it while the Activity runs.

    Register it on the Workers that run the Workflow and the Activities.
    """

    def intercept_activity(self, next: ActivityInboundInterceptor) -> ActivityInboundInterceptor:
        return _ActivityInbound(next)

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> type[WorkflowInboundInterceptor] | None:
        return _WorkflowInbound


class _WorkflowInbound(WorkflowInboundInterceptor):
    def init(self, outbound: WorkflowOutboundInterceptor) -> None:
        super().init(_WorkflowOutbound(outbound))


class _WorkflowOutbound(WorkflowOutboundInterceptor):
    def start_activity(self, input: StartActivityInput) -> workflow.ActivityHandle[Any]:
        context = _CURRENT_STREAM_CONTEXT.get()
        if context is not None:
            payload = PayloadConverter.default.to_payload(context.model_dump_json())
            input.headers = {**input.headers, _HEADER: payload}
        return super().start_activity(input)


class _ActivityInbound(ActivityInboundInterceptor):
    async def execute_activity(self, input: ExecuteActivityInput) -> Any:
        payload = input.headers.get(_HEADER)
        if payload is None:
            return await super().execute_activity(input)

        context = TurnStreamContext.model_validate_json(PayloadConverter.default.from_payload(payload, str))
        token = _CURRENT_STREAM_CONTEXT.set(context)
        try:
            return await super().execute_activity(input)
        finally:
            _CURRENT_STREAM_CONTEXT.reset(token)
