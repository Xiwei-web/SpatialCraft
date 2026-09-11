"""Provider-neutral model requests, responses, and runtime contracts."""

from __future__ import annotations

import asyncio
import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from typing import Any, TypeAlias

from spatialcraft.schemas._base import new_id, require_non_empty

JsonObject: TypeAlias = dict[str, Any]


class ModelError(RuntimeError):
    """Base class for model-layer failures."""


class ModelConfigurationError(ModelError):
    """Raised when a model/provider configuration is incomplete or invalid."""


class ModelCapabilityError(ModelError):
    """Raised when a request needs a capability the model does not advertise."""


class ModelRequestError(ModelError):
    """Raised when a provider rejects or cannot execute a request."""


class ModelResponseError(ModelError):
    """Raised when a provider response cannot be normalized safely."""


class MessageRole(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentPart:
    """One text or media part in a provider-neutral message."""

    kind: ContentKind
    text: str | None = None
    uri: str | None = None
    data: bytes | None = None
    mime_type: str | None = None
    detail: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.kind is ContentKind.TEXT:
            object.__setattr__(self, "text", require_non_empty(self.text or "", "text"))
            if self.uri is not None or self.data is not None:
                raise ValueError("text content cannot include uri or data")
            return
        if self.text is not None:
            raise ValueError("media content cannot include text")
        if (self.uri is None) == (self.data is None):
            raise ValueError("media content requires exactly one of uri or data")
        if self.uri is not None:
            object.__setattr__(self, "uri", require_non_empty(self.uri, "uri"))
        if self.data is not None:
            object.__setattr__(self, "data", bytes(self.data))
            if not self.data:
                raise ValueError("media data cannot be empty")
            object.__setattr__(
                self,
                "mime_type",
                require_non_empty(self.mime_type or "", "mime_type"),
            )

    @classmethod
    def text_part(cls, text: str) -> ContentPart:
        return cls(kind=ContentKind.TEXT, text=text)

    @classmethod
    def image_uri(
        cls, uri: str, *, mime_type: str | None = None, detail: str | None = None
    ) -> ContentPart:
        return cls(
            kind=ContentKind.IMAGE,
            uri=uri,
            mime_type=mime_type,
            detail=detail,
        )

    @classmethod
    def image_bytes(
        cls, data: bytes, *, mime_type: str = "image/png", detail: str | None = None
    ) -> ContentPart:
        return cls(
            kind=ContentKind.IMAGE,
            data=data,
            mime_type=mime_type,
            detail=detail,
        )

    def as_data_uri(self) -> str:
        """Return a media part as a URL, encoding bytes when necessary."""

        if self.kind is ContentKind.TEXT:
            raise TypeError("text content has no data URI")
        if self.uri is not None:
            return self.uri
        assert self.data is not None and self.mime_type is not None
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelMessage:
    """One normalized conversation message."""

    role: MessageRole
    content: tuple[ContentPart, ...]
    tool_calls: tuple[ResponseToolCall, ...] = ()
    name: str | None = None
    tool_call_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", tuple(self.content))
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.content and not (
            self.role is MessageRole.ASSISTANT and self.tool_calls
        ):
            raise ValueError("message content cannot be empty")
        if self.name is not None:
            object.__setattr__(self, "name", require_non_empty(self.name, "name"))
        if self.role is MessageRole.TOOL:
            object.__setattr__(
                self,
                "tool_call_id",
                require_non_empty(self.tool_call_id or "", "tool_call_id"),
            )
            if any(part.kind is not ContentKind.TEXT for part in self.content):
                raise ValueError("tool-result messages must contain text only")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError("only assistant messages may contain tool_calls")
        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("assistant tool calls must have unique call ids")

    @classmethod
    def text(cls, role: MessageRole, text: str, **kwargs: Any) -> ModelMessage:
        return cls(role=role, content=(ContentPart.text_part(text),), **kwargs)

    @property
    def text_content(self) -> str:
        return "\n".join(
            part.text or "" for part in self.content if part.kind is ContentKind.TEXT
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDefinition:
    """A callable tool described by JSON Schema."""

    name: str
    description: str
    parameters: JsonObject
    strict: bool = True
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_non_empty(self.name, "name"))
        object.__setattr__(
            self,
            "description",
            require_non_empty(self.description, "description"),
        )
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters must be a JSON Schema object")
        try:
            json.dumps(self.parameters, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("tool parameters must be JSON serializable") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationSettings:
    """Portable generation controls; providers ignore only documented extras."""

    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    reasoning_effort: str | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    response_format: JsonObject | None = None
    extra: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop", tuple(self.stop))
        object.__setattr__(self, "extra", dict(self.extra))
        if self.response_format is not None:
            object.__setattr__(self, "response_format", dict(self.response_format))
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if any(not item for item in self.stop):
            raise ValueError("stop sequences cannot be empty")
        if self.top_logprobs is not None and self.top_logprobs < 0:
            raise ValueError("top_logprobs cannot be negative")
        if self.top_logprobs and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs=True")

    def overlay(self, overrides: MappingLike | None) -> GenerationSettings:
        """Return settings with non-``None`` mapping values applied."""

        if not overrides:
            return self
        allowed = {field.name for field in self.__dataclass_fields__.values()}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"unknown generation settings: {sorted(unknown)}")
        values = {key: value for key, value in overrides.items() if value is not None}
        if "stop" in values:
            values["stop"] = tuple(values["stop"])
        return replace(self, **values)


MappingLike: TypeAlias = dict[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRequest:
    """Complete provider-neutral inference request."""

    model_alias: str
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()
    settings: GenerationSettings = field(default_factory=GenerationSettings)
    tool_choice: str | None = None
    parallel_tool_calls: bool = True
    request_id: str = field(default_factory=lambda: new_id("model_request"))
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_alias", require_non_empty(self.model_alias, "model_alias")
        )
        object.__setattr__(
            self, "request_id", require_non_empty(self.request_id, "request_id")
        )
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.messages:
            raise ValueError("model request requires at least one message")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique within a request")
        if self.tool_choice not in {None, "auto", "none", "required"} and (
            self.tool_choice not in set(names)
        ):
            raise ValueError("tool_choice must be a mode or a declared tool name")
        if self.tool_choice not in {None, "none"} and not self.tools:
            raise ValueError("tool_choice requires declared tools")


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseToolCall:
    """Normalized function call emitted by a model."""

    name: str
    arguments: JsonObject
    call_id: str = field(default_factory=lambda: new_id("call"))
    raw_arguments: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_non_empty(self.name, "name"))
        object.__setattr__(self, "call_id", require_non_empty(self.call_id, "call_id"))
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenUsage:
    """Provider-reported counts; absent usage is unknown, never a zero claim."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.reasoning_tokens,
            self.cached_input_tokens,
        )
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in values
        ):
            raise ValueError("token counts must be nonnegative integers or unknown")
        if (
            self.total_tokens is None
            and self.input_tokens is not None
            and self.output_tokens is not None
        ):
            object.__setattr__(
                self, "total_tokens", self.input_tokens + self.output_tokens
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponse:
    """Provider response normalized for the SpatialCraft agent loop."""

    provider: str
    model: str
    text: str | None = None
    tool_calls: tuple[ResponseToolCall, ...] = ()
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    response_id: str = field(default_factory=lambda: new_id("model_response"))
    raw: Any = None
    latency_ms: float | None = None
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("provider", "model", "response_id"):
            object.__setattr__(self, name, require_non_empty(getattr(self, name), name))
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.text is not None:
            object.__setattr__(self, "text", self.text.strip())
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("response tool calls must have unique call_id values")


@dataclass(frozen=True, slots=True, kw_only=True)
class SequenceScore:
    """Token log-probabilities for a fixed target continuation."""

    model: str
    target_text: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    prompt_token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", require_non_empty(self.model, "model"))
        object.__setattr__(
            self,
            "target_text",
            require_non_empty(self.target_text, "target_text"),
        )
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "token_logprobs", tuple(self.token_logprobs))
        if not self.token_ids or len(self.token_ids) != len(self.token_logprobs):
            raise ValueError(
                "token ids and log-probabilities must be non-empty and aligned"
            )
        if not all(isfinite(value) for value in self.token_logprobs):
            raise ValueError("token log-probabilities must be finite")
        if self.prompt_token_count < 0:
            raise ValueError("prompt_token_count cannot be negative")

    @property
    def mean_logprob(self) -> float:
        return sum(self.token_logprobs) / len(self.token_logprobs)


class ModelProvider(ABC):
    """Synchronous provider contract with an asynchronous thread wrapper."""

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Execute one inference request."""

    async def agenerate(self, request: ModelRequest) -> ModelResponse:
        return await asyncio.to_thread(self.generate, request)

    def score(self, request: ModelRequest, target_text: str) -> SequenceScore:
        raise ModelCapabilityError(
            f"{type(self).__name__} does not implement fixed-target scoring"
        )

    async def ascore(self, request: ModelRequest, target_text: str) -> SequenceScore:
        return await asyncio.to_thread(self.score, request, target_text)

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        raise ModelCapabilityError(
            f"{type(self).__name__} does not implement embeddings"
        )

    async def aembed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return await asyncio.to_thread(self.embed, texts)


__all__ = [
    "ContentKind",
    "ContentPart",
    "GenerationSettings",
    "JsonObject",
    "MessageRole",
    "ModelCapabilityError",
    "ModelConfigurationError",
    "ModelError",
    "ModelMessage",
    "ModelProvider",
    "ModelRequest",
    "ModelRequestError",
    "ModelResponse",
    "ModelResponseError",
    "ResponseToolCall",
    "SequenceScore",
    "TokenUsage",
    "ToolDefinition",
]
