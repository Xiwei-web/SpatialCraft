"""JSON-safe request/response audit codec (media bytes use base64, no API keys)."""

from __future__ import annotations

import base64
from dataclasses import asdict

from .interfaces import (
    ContentKind,
    ContentPart,
    GenerationSettings,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ResponseToolCall,
    TokenUsage,
    ToolDefinition,
)


def request_to_dict(request: ModelRequest, *, identity: bool = True) -> dict:
    data = asdict(request)
    if not identity:
        data.pop("request_id", None)
    for message in data["messages"]:
        for part in message["content"]:
            if part["data"] is not None:
                part["data"] = base64.b64encode(part["data"]).decode("ascii")
    return data


def request_from_dict(data: dict) -> ModelRequest:
    values = dict(data)
    messages = []
    for original in values.pop("messages"):
        msg = dict(original)
        parts = []
        for original_part in msg.pop("content"):
            part = dict(original_part)
            part["kind"] = ContentKind(part["kind"])
            if part.get("data") is not None:
                part["data"] = base64.b64decode(part["data"], validate=True)
            parts.append(ContentPart(**part))
        msg["role"] = MessageRole(msg["role"])
        msg["content"] = tuple(parts)
        msg["tool_calls"] = tuple(
            ResponseToolCall(**item) for item in msg.get("tool_calls", ())
        )
        messages.append(ModelMessage(**msg))
    values["messages"] = tuple(messages)
    values["settings"] = GenerationSettings(**values["settings"])
    values["tools"] = tuple(ToolDefinition(**tool) for tool in values.get("tools", ()))
    return ModelRequest(**values)


def response_to_dict(response: ModelResponse) -> dict:
    return asdict(response)


def response_from_dict(data: dict) -> ModelResponse:
    values = dict(data)
    values["usage"] = TokenUsage(**values["usage"])
    values["tool_calls"] = tuple(
        ResponseToolCall(**c) for c in values.get("tool_calls", ())
    )
    return ModelResponse(**values)
