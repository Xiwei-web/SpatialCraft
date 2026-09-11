"""OpenAI Responses API provider."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
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

    def parameter_audit(self, request: ModelRequest) -> dict[str, Any]:
        """Describe sent sampling parameters without loading a client or any media."""
        settings = request.settings
        requires_none = bool(
            re.fullmatch(r"gpt-5\.4(?:-\d{4}-\d{2}-\d{2})?", self.config.model_id)
            or self.config.metadata.get("sampling_requires_reasoning_none", False)
        )
        # An unspecified effort does not establish the required explicit none.
        omit_sampling = requires_none and settings.reasoning_effort != "none"
        return {
            "requested_temperature": settings.temperature,
            "requested_top_p": settings.top_p,
            "effective_temperature": None if omit_sampling else settings.temperature,
            "effective_top_p": None if omit_sampling else settings.top_p,
            "sampling_parameters_omitted": omit_sampling,
            "sampling_parameter_policy": (
                "omit_unsupported_reasoning_sampling"
                if omit_sampling
                else "send_requested"
            ),
            "reasoning_effort": settings.reasoning_effort,
            "effective_value_semantics": "null means parameter not sent; server default is not inferred",
        }

    @staticmethod
    def api_metadata(request: ModelRequest) -> dict[str, str]:
        """Send bounded lookup fields; the complete metadata remains in the journal."""
        metadata = request.metadata
        if not metadata:
            return {}
        encoded = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False
        )
        result = {"audit_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
        keys = (
            "protocol_version",
            "operation",
            "role",
            "phase",
            "task_id",
            "snapshot_id",
            "rollout_index",
            "step_index",
            "state_id",
            "retry_index",
            "template_sha256",
            "rendered_prompt_sha256",
            "input_snapshot_id",
        )
        for key in keys:
            value = metadata.get(key)
            if value is None or not isinstance(value, (str, int, float, bool)):
                continue
            text = str(value)
            result[key] = (
                text
                if len(text) <= 512
                else "sha256:" + hashlib.sha256(text.encode()).hexdigest()
            )
        if len(result) > 16 or any(
            len(k) > 64 or len(v) > 512 for k, v in result.items()
        ):
            raise ModelRequestError("API metadata exceeds Responses limits")
        return result

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
            if message.role is MessageRole.ASSISTANT:
                if any(part.kind is not ContentKind.TEXT for part in message.content):
                    raise ModelRequestError(
                        "Assistant history must contain output text"
                    )
                content = [
                    {"type": "output_text", "text": part.text, "annotations": []}
                    for part in message.content
                ]
            else:
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
        sampling = self.parameter_audit(request)
        if sampling["effective_temperature"] is not None:
            payload["temperature"] = sampling["effective_temperature"]
        if sampling["effective_top_p"] is not None:
            payload["top_p"] = sampling["effective_top_p"]
        if settings.reasoning_effort is not None:
            payload["reasoning"] = {"effort": settings.reasoning_effort}
        if settings.logprobs:
            if sampling["sampling_parameters_omitted"]:
                raise ModelRequestError(
                    "This reasoning mode does not support requested logprobs"
                )
            payload["top_logprobs"] = settings.top_logprobs or 0
        if settings.response_format is not None:
            payload["text"] = {"format": settings.response_format}
        metadata = self.api_metadata(request)
        if metadata:
            payload["metadata"] = metadata
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
        reserved = set(payload) | {
            "metadata",
            "temperature",
            "top_p",
            "reasoning",
            "seed",
            "top_logprobs",
            "logprobs",
        }
        overlap = reserved & set(settings.extra)
        if overlap:
            raise ModelRequestError(
                f"generation.extra cannot replace managed fields: {sorted(overlap)}"
            )
        payload.update(settings.extra)
        return payload

    def generate(self, request: ModelRequest) -> ModelResponse:
        payload = self.build_payload(request)
        client = self._make_client()
        started = perf_counter()
        try:
            response = client.responses.create(**payload)
        except Exception as exc:
            raise ModelRequestError(
                f"OpenAI Responses request failed for {self.config.alias}: {exc}"
            ) from exc
        latency_ms = (perf_counter() - started) * 1000
        parsed = parse_openai_responses(
            response,
            provider=self.provider_name,
            model=self.config.model_id,
            latency_ms=latency_ms,
        )
        return replace(
            parsed,
            raw={
                **(parsed.raw or {}),
                "spatialcraft_request_parameters": self.parameter_audit(request),
                "spatialcraft_api_metadata": payload.get("metadata", {}),
            },
        )


__all__ = ["OpenAIResponsesProvider"]
