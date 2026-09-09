"""Agent action and tool-call schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ._base import SchemaMixin, new_id, normalize_datetime, require_non_empty, utc_now


class ActionType(str, Enum):
    TOOL = "tool"
    FINAL = "final"
    NOOP = "noop"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall(SchemaMixin):
    """One validated request to a registered tool."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: new_id("call"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool_name", require_non_empty(self.tool_name, "tool_name")
        )
        object.__setattr__(self, "call_id", require_non_empty(self.call_id, "call_id"))
        object.__setattr__(self, "arguments", dict(self.arguments))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentAction(SchemaMixin):
    """A normalized model decision independent of provider response format."""

    action_type: ActionType
    action_id: str = field(default_factory=lambda: new_id("action"))
    tool_calls: tuple[ToolCall, ...] = ()
    final_answer: str | None = None
    reasoning_summary: str | None = None
    raw_response: dict[str, Any] | str | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "action_id", require_non_empty(self.action_id, "action_id")
        )
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "metadata", dict(self.metadata))

        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool_calls must have unique call_id values")
        if self.action_type is ActionType.TOOL:
            if not self.tool_calls:
                raise ValueError("tool actions require at least one tool call")
            if self.final_answer is not None:
                raise ValueError("tool actions cannot include a final answer")
        elif self.action_type is ActionType.FINAL:
            if self.tool_calls:
                raise ValueError("final actions cannot include tool calls")
            object.__setattr__(
                self,
                "final_answer",
                require_non_empty(self.final_answer or "", "final_answer"),
            )
        elif self.action_type is ActionType.NOOP:
            if self.tool_calls or self.final_answer is not None:
                raise ValueError("noop actions cannot contain tool calls or an answer")

    @classmethod
    def tool(
        cls, *tool_calls: ToolCall, reasoning_summary: str | None = None
    ) -> AgentAction:
        return cls(
            action_type=ActionType.TOOL,
            tool_calls=tuple(tool_calls),
            reasoning_summary=reasoning_summary,
        )

    @classmethod
    def final(cls, answer: str, *, reasoning_summary: str | None = None) -> AgentAction:
        return cls(
            action_type=ActionType.FINAL,
            final_answer=answer,
            reasoning_summary=reasoning_summary,
        )

    @classmethod
    def noop(cls, *, reason: str | None = None) -> AgentAction:
        return cls(action_type=ActionType.NOOP, reasoning_summary=reason)
