"""Policy checks around provider-neutral response-to-action parsing."""

from __future__ import annotations

from dataclasses import replace

from spatialcraft.models import ModelResponse, ModelResponseError, to_agent_action
from spatialcraft.schemas import ActionType, AgentAction
from spatialcraft.tools import ToolRegistry


class ActionParser:
    def __init__(self, tools: ToolRegistry, *, max_parallel_calls: int = 8) -> None:
        if max_parallel_calls < 1:
            raise ValueError("max_parallel_calls must be positive")
        self.tools = tools
        self.max_parallel_calls = max_parallel_calls

    def parse(self, response: ModelResponse) -> AgentAction:
        action = to_agent_action(response)
        if action.action_type is ActionType.TOOL:
            if len(action.tool_calls) > self.max_parallel_calls:
                raise ModelResponseError(
                    f"model emitted {len(action.tool_calls)} tool calls; "
                    f"limit is {self.max_parallel_calls}"
                )
            unknown = sorted(
                {call.tool_name for call in action.tool_calls} - set(self.tools.names())
            )
            if unknown:
                raise ModelResponseError(f"model requested unknown tools: {unknown}")
        raw = response.raw
        if raw is not None and not isinstance(raw, (dict, str)):
            raw = repr(raw)
        return replace(
            action,
            raw_response=raw,
            metadata={
                **action.metadata,
                "model": response.model,
                "provider": response.provider,
                "response_id": response.response_id,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            },
        )


__all__ = ["ActionParser"]
