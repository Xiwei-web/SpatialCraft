"""Normalized outputs produced by local or remote spatial tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Any

from ._base import (
    SchemaMixin,
    new_id,
    normalize_datetime,
    require_non_empty,
    utc_now,
)


class ToolStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class ArtifactType(str, Enum):
    IMAGE = "image"
    MASK = "mask"
    DEPTH = "depth"
    POINT_CLOUD = "point_cloud"
    BEV = "bev"
    BOUNDING_BOXES = "bounding_boxes"
    POSE = "pose"
    OPTICAL_FLOW = "optical_flow"
    SCENE_GRAPH = "scene_graph"
    TEXT = "text"
    JSON = "json"
    ARRAY = "array"
    OTHER = "other"


@dataclass(frozen=True, slots=True, kw_only=True)
class CoordinateFrame(SchemaMixin):
    """Coordinate-frame metadata for geometric artifacts.

    ``transform_to_parent`` is a row-major homogeneous 4x4 matrix.
    """

    frame_id: str
    parent_frame_id: str | None = None
    transform_to_parent: tuple[float, ...] | None = None
    unit: str = "pixel"
    convention: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", require_non_empty(self.frame_id, "frame_id")
        )
        object.__setattr__(self, "unit", require_non_empty(self.unit, "unit"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.parent_frame_id == self.frame_id:
            raise ValueError("a coordinate frame cannot be its own parent")
        if self.transform_to_parent is not None:
            transform = tuple(float(value) for value in self.transform_to_parent)
            if len(transform) != 16:
                raise ValueError("transform_to_parent must contain 16 values")
            if not all(isfinite(value) for value in transform):
                raise ValueError("transform_to_parent must contain finite values")
            object.__setattr__(self, "transform_to_parent", transform)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRef(SchemaMixin):
    """A reference to a persisted tool artifact, never the artifact bytes itself."""

    artifact_type: ArtifactType
    uri: str
    artifact_id: str = field(default_factory=lambda: new_id("artifact"))
    mime_type: str | None = None
    shape: tuple[int, ...] = ()
    dtype: str | None = None
    sha256: str | None = None
    frame_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", require_non_empty(self.uri, "uri"))
        object.__setattr__(
            self, "artifact_id", require_non_empty(self.artifact_id, "artifact_id")
        )
        shape = tuple(int(size) for size in self.shape)
        if any(size < 0 for size in shape):
            raise ValueError("artifact shape dimensions cannot be negative")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.sha256 is not None:
            digest = self.sha256.lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("sha256 must be a 64-character hexadecimal digest")
            object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResult(SchemaMixin):
    """Provider-neutral result of executing one :class:`ToolCall`."""

    tool_call_id: str
    tool_name: str
    status: ToolStatus
    result_id: str = field(default_factory=lambda: new_id("result"))
    text: str | None = None
    structured_output: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    coordinate_frames: tuple[CoordinateFrame, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime = field(default_factory=utc_now)
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("tool_call_id", "tool_name", "result_id"):
            object.__setattr__(self, name, require_non_empty(getattr(self, name), name))
        object.__setattr__(self, "structured_output", dict(self.structured_output))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "coordinate_frames", tuple(self.coordinate_frames))
        object.__setattr__(self, "started_at", normalize_datetime(self.started_at))
        object.__setattr__(self, "finished_at", normalize_datetime(self.finished_at))
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.status is ToolStatus.FAILED and not self.error_message:
            raise ValueError("failed tool results require error_message")
        if self.status is ToolStatus.SUCCEEDED and self.error_message:
            raise ValueError("successful tool results cannot contain error_message")

        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifacts must have unique artifact_id values")
        frame_ids = {frame.frame_id for frame in self.coordinate_frames}
        if len(frame_ids) != len(self.coordinate_frames):
            raise ValueError("coordinate_frames must have unique frame_id values")
        unknown_frames = {
            artifact.frame_id
            for artifact in self.artifacts
            if artifact.frame_id is not None and artifact.frame_id not in frame_ids
        }
        if unknown_frames:
            raise ValueError(
                f"artifacts reference undeclared coordinate frames: {sorted(unknown_frames)}"
            )

    @property
    def succeeded(self) -> bool:
        return self.status is ToolStatus.SUCCEEDED
