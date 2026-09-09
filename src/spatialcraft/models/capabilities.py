"""Model capability declarations and request compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .interfaces import ContentKind, ModelCapabilityError, ModelRequest


class Capability(str, Enum):
    TEXT_INPUT = "text_input"
    IMAGE_INPUT = "image_input"
    AUDIO_INPUT = "audio_input"
    VIDEO_INPUT = "video_input"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    REASONING = "reasoning"
    LOGPROBS = "logprobs"
    FIXED_TARGET_SCORING = "fixed_target_scoring"
    EMBEDDINGS = "embeddings"
    STREAMING = "streaming"


_CONTENT_CAPABILITY = {
    ContentKind.TEXT: Capability.TEXT_INPUT,
    ContentKind.IMAGE: Capability.IMAGE_INPUT,
    ContentKind.AUDIO: Capability.AUDIO_INPUT,
    ContentKind.VIDEO: Capability.VIDEO_INPUT,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapabilities:
    """Capabilities advertised by one concrete model configuration."""

    supported: frozenset[Capability] = field(
        default_factory=lambda: frozenset({Capability.TEXT_INPUT})
    )
    context_window: int | None = None
    max_output_tokens: int | None = None
    max_images: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported",
            frozenset(
                item if isinstance(item, Capability) else Capability(item)
                for item in self.supported
            ),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        for name in ("context_window", "max_output_tokens", "max_images"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        if Capability.TEXT_INPUT not in self.supported:
            raise ValueError("all configured language models must support text input")

    def supports(self, capability: Capability | str) -> bool:
        return Capability(capability) in self.supported

    def require(self, *capabilities: Capability | str) -> None:
        required = {Capability(item) for item in capabilities}
        missing = required - self.supported
        if missing:
            raise ModelCapabilityError(
                "model lacks required capabilities: "
                + ", ".join(sorted(item.value for item in missing))
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": sorted(item.value for item in self.supported),
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "max_images": self.max_images,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ModelCapabilities:
        data = dict(data or {})
        supported = data.pop("supported", [Capability.TEXT_INPUT.value])
        return cls(supported=frozenset(supported), **data)


def required_capabilities(request: ModelRequest) -> frozenset[Capability]:
    """Infer the minimum advertised capability set for a request."""

    required = {Capability.TEXT_INPUT}
    image_count = 0
    for message in request.messages:
        for part in message.content:
            required.add(_CONTENT_CAPABILITY[part.kind])
            image_count += int(part.kind is ContentKind.IMAGE)
    if request.tools:
        required.add(Capability.TOOL_CALLING)
    if request.settings.response_format is not None:
        required.add(Capability.STRUCTURED_OUTPUT)
    if request.settings.reasoning_effort is not None:
        required.add(Capability.REASONING)
    if request.settings.logprobs:
        required.add(Capability.LOGPROBS)
    return frozenset(required)


def validate_request_capabilities(
    request: ModelRequest, capabilities: ModelCapabilities
) -> None:
    """Raise a detailed error when a request exceeds a model declaration."""

    required = required_capabilities(request)
    missing = required - capabilities.supported
    if missing:
        raise ModelCapabilityError(
            f"request {request.request_id} requires unsupported capabilities: "
            + ", ".join(sorted(item.value for item in missing))
        )
    requested_output = request.settings.max_output_tokens
    if (
        requested_output is not None
        and capabilities.max_output_tokens is not None
        and requested_output > capabilities.max_output_tokens
    ):
        raise ModelCapabilityError(
            f"requested {requested_output} output tokens exceeds model limit "
            f"{capabilities.max_output_tokens}"
        )
    if capabilities.max_images is not None:
        image_count = sum(
            part.kind is ContentKind.IMAGE
            for message in request.messages
            for part in message.content
        )
        if image_count > capabilities.max_images:
            raise ModelCapabilityError(
                f"request contains {image_count} images; model limit is "
                f"{capabilities.max_images}"
            )


__all__ = [
    "Capability",
    "ModelCapabilities",
    "required_capabilities",
    "validate_request_capabilities",
]
