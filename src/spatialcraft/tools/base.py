"""Contracts shared by local and remote spatial-tool implementations."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from spatialcraft.models import ToolDefinition
from spatialcraft.schemas import ArtifactType, CoordinateFrame, ToolStatus
from spatialcraft.schemas._base import new_id, require_non_empty, require_probability

JsonObject: TypeAlias = dict[str, Any]


class ToolError(RuntimeError):
    """Base error raised by tool infrastructure and implementations."""


class ToolConfigurationError(ToolError):
    """Raised when a tool or backend is configured incorrectly."""


class ToolExecutionError(ToolError):
    """Raised when a tool cannot complete a valid invocation."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolSpec:
    """Stable public contract presented to an agent model."""

    name: str
    description: str
    input_schema: JsonObject
    output_schema: JsonObject | None = None
    version: str = "1.0.0"
    default_timeout_s: float = 60.0
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("name", "description", "version"):
            object.__setattr__(self, name, require_non_empty(getattr(self, name), name))
        object.__setattr__(self, "input_schema", dict(self.input_schema))
        if self.output_schema is not None:
            object.__setattr__(self, "output_schema", dict(self.output_schema))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.input_schema.get("type") != "object":
            raise ValueError("tool input_schema must describe a JSON object")
        if self.default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be positive")

    def as_model_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_schema,
            strict=True,
            metadata={"version": self.version, **self.metadata},
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolContext:
    """Invocation-scoped identifiers and non-secret execution metadata."""

    run_id: str
    invocation_id: str = field(default_factory=lambda: new_id("tool_invocation"))
    task_id: str | None = None
    state_id: str | None = None
    timeout_s: float | None = None
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", require_non_empty(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "invocation_id",
            require_non_empty(self.invocation_id, "invocation_id"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    def to_wire(self) -> JsonObject:
        return {
            "run_id": self.run_id,
            "invocation_id": self.invocation_id,
            "task_id": self.task_id,
            "state_id": self.state_id,
            "timeout_s": self.timeout_s,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactPayload:
    """An unpersisted artifact produced by a tool backend."""

    artifact_type: ArtifactType
    data: bytes | None = None
    text: str | None = None
    json_value: Any | None = None
    source_path: str | Path | None = None
    suffix: str = ""
    mime_type: str | None = None
    shape: tuple[int, ...] = ()
    dtype: str | None = None
    frame_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        sources = (
            self.data is not None,
            self.text is not None,
            self.json_value is not None,
            self.source_path is not None,
        )
        if sum(sources) != 1:
            raise ValueError("artifact payload requires exactly one content source")
        if self.data is not None:
            object.__setattr__(self, "data", bytes(self.data))
            if not self.data:
                raise ValueError("artifact bytes cannot be empty")
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if any(value < 0 for value in self.shape):
            raise ValueError("artifact shape dimensions cannot be negative")
        if self.suffix and (
            not self.suffix.startswith(".") or "/" in self.suffix or "\\" in self.suffix
        ):
            raise ValueError("artifact suffix must be empty or a file extension")

    def to_wire(self) -> JsonObject:
        if self.source_path is not None:
            data = Path(self.source_path).read_bytes()
            content_kind, content = "data_b64", base64.b64encode(data).decode("ascii")
        elif self.data is not None:
            content_kind = "data_b64"
            content = base64.b64encode(self.data).decode("ascii")
        elif self.text is not None:
            content_kind, content = "text", self.text
        else:
            content_kind, content = "json_value", self.json_value
        return {
            "artifact_type": self.artifact_type.value,
            content_kind: content,
            "suffix": self.suffix,
            "mime_type": self.mime_type,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "frame_id": self.frame_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> ArtifactPayload:
        payload = dict(value)
        if "data_b64" in payload:
            payload["data"] = base64.b64decode(payload.pop("data_b64"), validate=True)
        payload["artifact_type"] = ArtifactType(payload["artifact_type"])
        payload["shape"] = tuple(payload.get("shape") or ())
        return cls(**payload)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecution:
    """Backend result before artifacts receive durable references."""

    status: ToolStatus = ToolStatus.SUCCEEDED
    text: str | None = None
    structured_output: JsonObject = field(default_factory=dict)
    artifacts: tuple[ArtifactPayload, ...] = ()
    coordinate_frames: tuple[CoordinateFrame, ...] = ()
    confidence: float | None = None
    unit: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "structured_output", dict(self.structured_output))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "coordinate_frames", tuple(self.coordinate_frames))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                require_probability(self.confidence, "confidence"),
            )
        if self.status is ToolStatus.FAILED and not self.error_message:
            raise ValueError("failed executions require error_message")
        if self.status is ToolStatus.SUCCEEDED and self.error_message:
            raise ValueError("successful executions cannot have error_message")
        frame_ids = {frame.frame_id for frame in self.coordinate_frames}
        if len(frame_ids) != len(self.coordinate_frames):
            raise ValueError("coordinate frame ids must be unique")
        missing = {
            item.frame_id
            for item in self.artifacts
            if item.frame_id is not None and item.frame_id not in frame_ids
        }
        if missing:
            raise ValueError(f"artifacts reference unknown frames: {sorted(missing)}")

    def to_wire(self) -> JsonObject:
        return {
            "status": self.status.value,
            "text": self.text,
            "structured_output": dict(self.structured_output),
            "artifacts": [item.to_wire() for item in self.artifacts],
            "coordinate_frames": [frame.to_dict() for frame in self.coordinate_frames],
            "confidence": self.confidence,
            "unit": self.unit,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> ToolExecution:
        payload = dict(value)
        payload["status"] = ToolStatus(payload.get("status", ToolStatus.SUCCEEDED))
        payload["artifacts"] = tuple(
            ArtifactPayload.from_wire(item) for item in payload.get("artifacts", ())
        )
        payload["coordinate_frames"] = tuple(
            CoordinateFrame.from_dict(item)
            for item in payload.get("coordinate_frames", ())
        )
        return cls(**payload)


class SpatialTool(ABC):
    """Executable implementation behind one stable :class:`ToolSpec`."""

    spec: ToolSpec

    @abstractmethod
    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        """Execute validated arguments and return unpersisted outputs."""


class FunctionTool(SpatialTool):
    """Wrap a plain callable as a local spatial tool."""

    def __init__(
        self,
        spec: ToolSpec,
        function: Callable[[Mapping[str, Any], ToolContext], ToolExecution],
    ) -> None:
        self.spec = spec
        self.function = function

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        return self.function(arguments, context)


__all__ = [
    "ArtifactPayload",
    "FunctionTool",
    "JsonObject",
    "SpatialTool",
    "ToolConfigurationError",
    "ToolContext",
    "ToolError",
    "ToolExecution",
    "ToolExecutionError",
    "ToolSpec",
]
