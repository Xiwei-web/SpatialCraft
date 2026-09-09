"""OpenAI Chat Completions compatible provider."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

from ..capabilities import validate_request_capabilities
from ..interfaces import (
    ContentKind,
    MessageRole,
    ModelConfigurationError,
    ModelProvider,
    ModelRequest,
    ModelRequestError,
    ModelResponse,
)
from ..registry import ModelConfig, ProviderKind
from ..response_parser import parse_openai_chat_response


class OpenAIChatCompatibleProvider(ModelProvider):
    """Call OpenAI-compatible ``/v1/chat/completions`` endpoints."""

    provider_name = ProviderKind.OPENAI_CHAT_COMPATIBLE.value
    accepted_provider_kinds = frozenset(
        {ProviderKind.OPENAI_CHAT_COMPATIBLE, ProviderKind.VLLM_CLIENT}
    )

    def __init__(self, config: ModelConfig, *, client: Any | None = None) -> None:
        if config.provider not in self.accepted_provider_kinds:
            raise ModelConfigurationError(
                f"{type(self).__name__} cannot serve {config.provider.value}"
            )
        self.config = config
        self._client = client

    def _api_key_required(self) -> bool:
        return True

    def _default_api_key(self) -> str | None:
        return None

    def _make_client(self) -> Any:
        if self._client is not None:
            return self._client
        assert self.config.api is not None
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelConfigurationError(
                "install the 'openai' package for chat-compatible providers"
            ) from exc
        api_key = self.config.api.api_key(required=self._api_key_required())
        api_key = api_key or self._default_api_key()
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.config.api.timeout_seconds,
            "max_retries": self.config.api.max_retries,
        }
        base_url = self.config.api.resolved_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        if self.config.api.organization_env:
            organization = os.environ.get(self.config.api.organization_env)
            if organization:
                kwargs["organization"] = organization
        if self.config.api.default_headers:
            kwargs["default_headers"] = self.config.api.default_headers
        self._client = OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _message_content(message: Any) -> str | list[dict[str, Any]] | None:
        if not message.content:
            return None
        if all(part.kind is ContentKind.TEXT for part in message.content):
            return message.text_content
        parts: list[dict[str, Any]] = []
        for part in message.content:
            if part.kind is ContentKind.TEXT:
                parts.append({"type": "text", "text": part.text})
            elif part.kind is ContentKind.IMAGE:
                image: dict[str, Any] = {"url": part.as_data_uri()}
                if part.detail:
                    image["detail"] = part.detail
                parts.append({"type": "image_url", "image_url": image})
            else:
                raise ModelRequestError(
                    f"chat-compatible adapter does not encode {part.kind.value} input"
                )
        return parts

    def _messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role is MessageRole.TOOL:
                item: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.text_content,
                }
                if message.name:
                    item["name"] = message.name
                result.append(item)
                continue
            role = (
                "system"
                if message.role is MessageRole.DEVELOPER
                else message.role.value
            )
            item = {
                "role": role,
                "content": self._message_content(message),
            }
            if message.name:
                item["name"] = message.name
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.raw_arguments
                            or json.dumps(
                                call.arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                        },
                    }
                    for call in message.tool_calls
                ]
            result.append(item)
        return result

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        if request.model_alias != self.config.alias:
            raise ModelRequestError(
                f"request targets {request.model_alias}, provider serves {self.config.alias}"
            )
        validate_request_capabilities(request, self.config.capabilities)
        settings = request.settings
        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": self._messages(request),
        }
        if settings.max_output_tokens is not None:
            payload["max_tokens"] = settings.max_output_tokens
        if settings.temperature is not None:
            payload["temperature"] = settings.temperature
        if settings.top_p is not None:
            payload["top_p"] = settings.top_p
        if settings.stop:
            payload["stop"] = list(settings.stop)
        if settings.seed is not None:
            payload["seed"] = settings.seed
        if settings.reasoning_effort is not None:
            payload["reasoning_effort"] = settings.reasoning_effort
        if settings.logprobs:
            payload["logprobs"] = True
            if settings.top_logprobs is not None:
                payload["top_logprobs"] = settings.top_logprobs
        if settings.response_format is not None:
            payload["response_format"] = settings.response_format
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    },
                }
                for tool in request.tools
            ]
            payload["parallel_tool_calls"] = request.parallel_tool_calls
            if request.tool_choice in {None, "auto", "none", "required"}:
                if request.tool_choice is not None:
                    payload["tool_choice"] = request.tool_choice
            else:
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": request.tool_choice},
                }
        overlap = set(payload) & set(settings.extra)
        if overlap:
            raise ModelRequestError(
                f"generation.extra cannot replace managed fields: {sorted(overlap)}"
            )
        payload.update(settings.extra)
        return payload

    def generate(self, request: ModelRequest) -> ModelResponse:
        client = self._make_client()
        payload = self.build_payload(request)
        started = perf_counter()
        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:
            raise ModelRequestError(
                f"chat-completion request failed for {self.config.alias}: {exc}"
            ) from exc
        latency_ms = (perf_counter() - started) * 1000
        return parse_openai_chat_response(
            response,
            provider=self.provider_name,
            model=self.config.model_id,
            latency_ms=latency_ms,
        )


__all__ = ["OpenAIChatCompatibleProvider"]
