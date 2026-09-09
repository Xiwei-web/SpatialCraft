"""Native Google GenAI provider for Gemini models."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import unquote, urlparse

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
from ..response_parser import parse_gemini_response


class GeminiNativeProvider(ModelProvider):
    """Execute normalized requests with the official ``google-genai`` SDK."""

    provider_name = ProviderKind.GEMINI_NATIVE.value

    def __init__(self, config: ModelConfig, *, client: Any | None = None) -> None:
        if config.provider is not ProviderKind.GEMINI_NATIVE:
            raise ModelConfigurationError(
                f"GeminiNativeProvider cannot serve {config.provider.value}"
            )
        self.config = config
        self._client = client

    @staticmethod
    def _types() -> Any:
        try:
            from google.genai import types
        except ImportError as exc:
            raise ModelConfigurationError(
                "install the 'google-genai' package for the Gemini provider"
            ) from exc
        return types

    def _make_client(self) -> Any:
        if self._client is not None:
            return self._client
        assert self.config.api is not None
        try:
            from google import genai
        except ImportError as exc:
            raise ModelConfigurationError(
                "install the 'google-genai' package for the Gemini provider"
            ) from exc
        types = self._types()
        base_url = self.config.api.resolved_base_url()
        http_options = types.HttpOptions(
            base_url=base_url,
            headers=self.config.api.default_headers or None,
            timeout=int(self.config.api.timeout_seconds * 1000),
        )
        self._client = genai.Client(
            api_key=self.config.api.api_key(), http_options=http_options
        )
        return self._client

    @staticmethod
    def _local_path(uri: str) -> Path | None:
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if parsed.scheme == "":
            candidate = Path(uri).expanduser()
            return candidate if candidate.is_file() else None
        return None

    def _part(self, part: Any) -> Any:
        types = self._types()
        if part.kind is ContentKind.TEXT:
            return types.Part(text=part.text)
        if part.data is not None:
            return types.Part.from_bytes(data=part.data, mime_type=part.mime_type)
        assert part.uri is not None
        local = self._local_path(part.uri)
        if local is not None:
            mime_type = part.mime_type or "application/octet-stream"
            return types.Part.from_bytes(data=local.read_bytes(), mime_type=mime_type)
        return types.Part.from_uri(file_uri=part.uri, mime_type=part.mime_type)

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        if request.model_alias != self.config.alias:
            raise ModelRequestError(
                f"request targets {request.model_alias}, provider serves {self.config.alias}"
            )
        validate_request_capabilities(request, self.config.capabilities)
        types = self._types()
        system: list[str] = []
        contents: list[Any] = []
        for message in request.messages:
            if message.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}:
                system.append(message.text_content)
                continue
            if message.role is MessageRole.TOOL:
                if not message.name:
                    raise ModelRequestError(
                        "Gemini tool-result messages require the tool name"
                    )
                parts = [
                    types.Part.from_function_response(
                        name=message.name,
                        response={"output": message.text_content},
                    )
                ]
                role = "user"
            else:
                parts = [self._part(part) for part in message.content]
                parts.extend(
                    types.Part.from_function_call(
                        name=call.name,
                        args=call.arguments,
                    )
                    for call in message.tool_calls
                )
                role = "model" if message.role is MessageRole.ASSISTANT else "user"
            contents.append(types.Content(role=role, parts=parts))

        settings = request.settings
        config_kwargs: dict[str, Any] = {
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            )
        }
        if system:
            config_kwargs["system_instruction"] = "\n\n".join(system)
        if settings.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = settings.max_output_tokens
        if settings.temperature is not None:
            config_kwargs["temperature"] = settings.temperature
        if settings.top_p is not None:
            config_kwargs["top_p"] = settings.top_p
        if settings.stop:
            config_kwargs["stop_sequences"] = list(settings.stop)
        if settings.seed is not None:
            config_kwargs["seed"] = settings.seed
        if settings.logprobs:
            config_kwargs["response_logprobs"] = True
            if settings.top_logprobs is not None:
                config_kwargs["logprobs"] = settings.top_logprobs
        if settings.reasoning_effort is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=settings.reasoning_effort
            )
        if settings.response_format is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = settings.response_format
        if request.tools:
            declarations = [
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.parameters,
                )
                for tool in request.tools
            ]
            config_kwargs["tools"] = [types.Tool(function_declarations=declarations)]
            mode = request.tool_choice or "auto"
            if mode == "none":
                function_mode = "NONE"
                allowed = None
            elif mode in {"auto", "required"}:
                function_mode = "AUTO" if mode == "auto" else "ANY"
                allowed = None
            else:
                function_mode = "ANY"
                allowed = [mode]
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=function_mode, allowed_function_names=allowed
                )
            )
        overlap = set(config_kwargs) & set(settings.extra)
        if overlap:
            raise ModelRequestError(
                f"generation.extra cannot replace managed fields: {sorted(overlap)}"
            )
        config_kwargs.update(settings.extra)
        return {
            "model": self.config.model_id,
            "contents": contents,
            "config": types.GenerateContentConfig(**config_kwargs),
        }

    def generate(self, request: ModelRequest) -> ModelResponse:
        client = self._make_client()
        payload = self.build_payload(request)
        started = perf_counter()
        try:
            response = client.models.generate_content(**payload)
        except Exception as exc:
            raise ModelRequestError(
                f"Gemini request failed for {self.config.alias}: {exc}"
            ) from exc
        latency_ms = (perf_counter() - started) * 1000
        return parse_gemini_response(
            response,
            provider=self.provider_name,
            model=self.config.model_id,
            latency_ms=latency_ms,
        )


__all__ = ["GeminiNativeProvider"]
