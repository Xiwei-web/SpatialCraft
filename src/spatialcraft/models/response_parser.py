"""Normalize provider responses and convert them into agent actions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from spatialcraft.schemas import AgentAction, ToolCall
from spatialcraft.schemas._base import new_id

from .interfaces import (
    ModelResponse,
    ModelResponseError,
    ResponseToolCall,
    TokenUsage,
)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _dump_raw(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                continue
    return repr(value)


def parse_tool_arguments(value: Any) -> tuple[dict[str, Any], str | None]:
    """Parse provider tool arguments while preserving their raw representation."""

    if value is None:
        return {}, None
    if isinstance(value, Mapping):
        return dict(value), None
    if isinstance(value, str):
        raw = value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(
                f"tool arguments are not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ModelResponseError("tool arguments must decode to a JSON object")
        return parsed, raw
    raise ModelResponseError(f"unsupported tool-argument type: {type(value).__name__}")


def _usage(
    value: Any,
    *,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    total_names: tuple[str, ...] = ("total_tokens",),
) -> TokenUsage:
    def first(names: tuple[str, ...], default: int | None = None) -> int | None:
        for name in names:
            item = _get(value, name)
            if item is not None:
                return int(item)
        return default

    details = _get(value, "output_tokens_details") or _get(
        value, "completion_tokens_details"
    )
    input_details = _get(value, "input_tokens_details") or _get(
        value, "prompt_tokens_details"
    )
    reasoning = _get(details, "reasoning_tokens")
    cached = _get(input_details, "cached_tokens")
    if cached is None:
        cached = _get(value, "cached_content_token_count")
    return TokenUsage(
        input_tokens=first(input_names),
        output_tokens=first(output_names),
        total_tokens=first(total_names),
        reasoning_tokens=int(reasoning) if reasoning is not None else None,
        cached_input_tokens=int(cached) if cached is not None else None,
    )


def parse_openai_responses(
    response: Any, *, provider: str, model: str, latency_ms: float | None = None
) -> ModelResponse:
    """Parse an OpenAI Responses API object or equivalent mapping."""

    text = _get(response, "output_text")
    calls: list[ResponseToolCall] = []
    text_fragments: list[str] = []
    for item in _as_list(_get(response, "output", [])):
        item_type = _get(item, "type")
        if item_type == "function_call":
            arguments, raw = parse_tool_arguments(_get(item, "arguments"))
            calls.append(
                ResponseToolCall(
                    call_id=_get(item, "call_id") or _get(item, "id") or new_id("call"),
                    name=_get(item, "name"),
                    arguments=arguments,
                    raw_arguments=raw,
                )
            )
        elif item_type == "message":
            for content in _as_list(_get(item, "content", [])):
                if _get(content, "type") in {"output_text", "text"}:
                    fragment = _get(content, "text")
                    if fragment:
                        text_fragments.append(str(fragment))
    if not text and text_fragments:
        text = "\n".join(text_fragments)
    usage_value = _get(response, "usage")
    usage = _usage(
        usage_value,
        input_names=("input_tokens",),
        output_names=("output_tokens",),
    )
    return ModelResponse(
        provider=provider,
        model=_get(response, "model") or model,
        text=text,
        tool_calls=tuple(calls),
        finish_reason=_get(response, "status"),
        usage=usage,
        response_id=_get(response, "id") or new_id("model_response"),
        raw=_dump_raw(response),
        latency_ms=latency_ms,
    )


def parse_openai_chat_response(
    response: Any, *, provider: str, model: str, latency_ms: float | None = None
) -> ModelResponse:
    """Parse OpenAI Chat Completions and compatible server responses."""

    choices = _as_list(_get(response, "choices", []))
    if not choices:
        raise ModelResponseError("chat completion contains no choices")
    choice = choices[0]
    message = _get(choice, "message")
    if message is None:
        raise ModelResponseError("chat completion choice contains no message")
    content = _get(message, "content")
    if isinstance(content, list):
        content = "\n".join(
            str(_get(part, "text"))
            for part in content
            if _get(part, "text") is not None
        )
    calls: list[ResponseToolCall] = []
    for item in _as_list(_get(message, "tool_calls", [])):
        function = _get(item, "function", item)
        arguments, raw = parse_tool_arguments(_get(function, "arguments"))
        calls.append(
            ResponseToolCall(
                call_id=_get(item, "id") or new_id("call"),
                name=_get(function, "name"),
                arguments=arguments,
                raw_arguments=raw,
            )
        )
    usage = _usage(
        _get(response, "usage"),
        input_names=("prompt_tokens", "input_tokens"),
        output_names=("completion_tokens", "output_tokens"),
    )
    return ModelResponse(
        provider=provider,
        model=_get(response, "model") or model,
        text=content,
        tool_calls=tuple(calls),
        finish_reason=_get(choice, "finish_reason"),
        usage=usage,
        response_id=_get(response, "id") or new_id("model_response"),
        raw=_dump_raw(response),
        latency_ms=latency_ms,
    )


def parse_gemini_response(
    response: Any, *, provider: str, model: str, latency_ms: float | None = None
) -> ModelResponse:
    """Parse a ``google.genai`` GenerateContent response."""

    calls: list[ResponseToolCall] = []
    fragments: list[str] = []
    finish_reason = None
    for candidate in _as_list(_get(response, "candidates", [])):
        finish_reason = (
            finish_reason
            or _get(candidate, "finish_reason")
            or _get(candidate, "finishReason")
        )
        content = _get(candidate, "content")
        for part in _as_list(_get(content, "parts", [])):
            text = _get(part, "text")
            if text and not _get(part, "thought", False):
                fragments.append(str(text))
            function_call = _get(part, "function_call") or _get(part, "functionCall")
            if function_call is not None:
                arguments, raw = parse_tool_arguments(
                    _get(function_call, "args", _get(function_call, "arguments"))
                )
                calls.append(
                    ResponseToolCall(
                        call_id=_get(function_call, "id") or new_id("call"),
                        name=_get(function_call, "name"),
                        arguments=arguments,
                        raw_arguments=raw,
                    )
                )
        break
    if not fragments:
        try:
            response_text = _get(response, "text")
        except (AttributeError, ValueError):
            response_text = None
        if response_text:
            fragments.append(str(response_text))
    usage_value = _get(response, "usage_metadata") or _get(response, "usageMetadata")
    usage = _usage(
        usage_value,
        input_names=("prompt_token_count", "promptTokenCount"),
        output_names=("candidates_token_count", "candidatesTokenCount"),
        total_names=("total_token_count", "totalTokenCount"),
    )
    return ModelResponse(
        provider=provider,
        model=model,
        text="\n".join(fragments) or None,
        tool_calls=tuple(calls),
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        usage=usage,
        response_id=_get(response, "response_id")
        or _get(response, "responseId")
        or new_id("model_response"),
        raw=_dump_raw(response),
        latency_ms=latency_ms,
    )


def parse_generated_text(
    text: str,
    *,
    provider: str,
    model: str,
    latency_ms: float | None = None,
    raw: Any = None,
) -> ModelResponse:
    """Normalize local generation and recognize an explicit tool-call envelope."""

    stripped = text.strip()
    if "</think>" in stripped:
        stripped = stripped.rsplit("</think>", 1)[1].strip()
    elif "<think>" in stripped:
        raise ModelResponseError(
            "Unfinished thinking must not become a final answer or tool call"
        )
    if "<tool_call>" in stripped:
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", stripped, re.DOTALL)
        if len(blocks) != stripped.count("<tool_call>"):
            raise ModelResponseError("Incomplete Qwen tool-call block")
        parsed_calls = []
        for block in blocks:
            match = re.fullmatch(
                r"<function=([\w.-]+)>\s*(.*?)\s*</function>", block, re.DOTALL
            )
            if match is None:
                raise ModelResponseError("Invalid Qwen function block")
            arguments = {}
            body = match.group(2)
            parameters = list(
                re.finditer(
                    r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL
                )
            )
            remainder = re.sub(
                r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>",
                "",
                body,
                flags=re.DOTALL,
            )
            if remainder.strip():
                raise ModelResponseError("Invalid Qwen parameter block")
            for parameter in parameters:
                name, value = parameter.group(1), parameter.group(2).strip()
                if name in arguments:
                    raise ModelResponseError("Duplicate Qwen parameter")
                try:
                    arguments[name] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[name] = value
            parsed_calls.append(
                ResponseToolCall(name=match.group(1), arguments=arguments)
            )
        return ModelResponse(
            provider=provider,
            model=model,
            text=stripped.split("<tool_call>", 1)[0].strip() or None,
            tool_calls=tuple(parsed_calls),
            finish_reason="stop",
            raw=raw if raw is not None else text,
            latency_ms=latency_ms,
        )
    calls: list[ResponseToolCall] = []
    answer: str | None = stripped or None
    try:
        envelope = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        envelope = None
    if isinstance(envelope, dict) and isinstance(envelope.get("tool_calls"), list):
        calls = []
        for item in envelope["tool_calls"]:
            if not isinstance(item, Mapping):
                raise ModelResponseError("local tool_calls entries must be objects")
            function = item.get("function", item)
            if not isinstance(function, Mapping):
                raise ModelResponseError("local tool call function must be an object")
            arguments, raw_arguments = parse_tool_arguments(function.get("arguments"))
            calls.append(
                ResponseToolCall(
                    call_id=item.get("id") or new_id("call"),
                    name=function.get("name"),
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )
        answer = envelope.get("text")
    return ModelResponse(
        provider=provider,
        model=model,
        text=answer,
        tool_calls=tuple(calls),
        finish_reason="stop",
        raw=raw if raw is not None else text,
        latency_ms=latency_ms,
    )


def to_agent_action(response: ModelResponse) -> AgentAction:
    """Convert a normalized model response to the execution schema."""

    if response.tool_calls:
        calls = tuple(
            ToolCall(
                tool_name=item.name,
                arguments=item.arguments,
                call_id=item.call_id,
                metadata={"model": response.model, "provider": response.provider},
            )
            for item in response.tool_calls
        )
        return AgentAction.tool(*calls, reasoning_summary=response.text or None)
    if response.text:
        return AgentAction.final(response.text)
    raise ModelResponseError("model response contains neither text nor tool calls")


__all__ = [
    "parse_gemini_response",
    "parse_generated_text",
    "parse_openai_chat_response",
    "parse_openai_responses",
    "parse_tool_arguments",
    "to_agent_action",
]
