"""OpenAI Responses API provider."""

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
from ..response_parser import parse_openai_responses


class OpenAIResponsesProvider(ModelProvider):
    """Execute normalized requests through ``client.responses.create``."""

    provider_name = ProviderKind.OPENAI_RESPONSES.value

    def __init__(self, config: ModelConfig, *, client: Any | None = None) -> None:
        if config.provider is not ProviderKind.OPENAI_RESPONSES:
            raise ModelConfigurationError(
                f"OpenAIResponsesProvider cannot serve {config.provider.value}"
            )
        self.config = config
        self._client = client

    def _make_client(self) -> Any:
        if self._client is not None:
            return self._client
        assert self.config.api is not None
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelConfigurationError(
                "install the 'openai' package for the Responses provider"
            ) from exc
        kwargs: dict[str, Any] = {
            "api_key": self.config.api.api_key(),
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
        if self.config.api.project_env:
            project = os.environ.get(self.config.api.project_env)
            if project:
                kwargs["project"] = project
        if self.config.api.default_headers:
            kwargs["default_headers"] = self.config.api.default_headers
        self._client = OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _content_part(part: Any) -> dict[str, Any]:
        if part.kind is ContentKind.TEXT:
            return {"type": "input_text", "text": part.text}
        if part.kind is ContentKind.IMAGE:
            result = {"type": "input_image", "image_url": part.as_data_uri()}
            if part.detail:
                result["detail"] = part.detail
            return result
        raise ModelRequestError(
            f"OpenAI Responses adapter does not encode {part.kind.value} input"
        )

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        if request.model_alias != self.config.alias:
            raise ModelRequestError(
                f"request targets {request.model_alias}, provider serves {self.config.alias}"
            )
        validate_request_capabilities(request, self.config.capabilities)
        instructions: list[str] = []
        inputs: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}:
                instructions.append(message.text_content)
                continue
            if message.role is MessageRole.TOOL:
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.text_content,
                    }
                )
                continue
            content = [self._content_part(part) for part in message.content]
            if content:
                inputs.append(
                    {
                        "role": message.role.value,
                        "content": content,
                    }
                )
            for call in message.tool_calls:
                inputs.append(
                    {
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.raw_arguments
                        or json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    }
                )

        settings = request.settings
        if settings.stop:
            raise ModelRequestError("Responses API does not accept stop sequences")
        if settings.seed is not None:
            raise ModelRequestError("Responses API does not accept a seed parameter")
        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "input": inputs,
            "parallel_tool_calls": request.parallel_tool_calls,
        }
        if instructions:
            payload["instructions"] = "\n\n".join(instructions)
        if settings.max_output_tokens is not None:
            payload["max_output_tokens"] = settings.max_output_tokens
        if settings.temperature is not None:
            payload["temperature"] = settings.temperature
        if settings.top_p is not None:
            payload["top_p"] = settings.top_p
        if settings.reasoning_effort is not None:
            payload["reasoning"] = {"effort": settings.reasoning_effort}
        if settings.logprobs:
            payload["top_logprobs"] = settings.top_logprobs or 0
        if settings.response_format is not None:
            payload["text"] = {"format": settings.response_format}
        if request.metadata:
            payload["metadata"] = {
                str(key): str(value) for key, value in request.metadata.items()
            }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }
                for tool in request.tools
            ]
            if request.tool_choice in {None, "auto", "none", "required"}:
                if request.tool_choice is not None:
                    payload["tool_choice"] = request.tool_choice
            else:
                payload["tool_choice"] = {
                    "type": "function",
                    "name": request.tool_choice,
                }
        reserved = set(payload)
        overlap = reserved & set(settings.extra)
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
            response = client.responses.create(**payload)
        except Exception as exc:
            raise ModelRequestError(
                f"OpenAI Responses request failed for {self.config.alias}: {exc}"
            ) from exc
        latency_ms = (perf_counter() - started) * 1000
        return parse_openai_responses(
            response,
            provider=self.provider_name,
            model=self.config.model_id,
            latency_ms=latency_ms,
        )


__all__ = ["OpenAIResponsesProvider"]
