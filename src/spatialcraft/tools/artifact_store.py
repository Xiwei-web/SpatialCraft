"""Content-addressed storage for every spatial-tool output and result audit."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from spatialcraft.schemas import ArtifactRef, ArtifactType, ToolResult
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import (
    atomic_copy,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
)

from .base import ArtifactPayload

_DEFAULT_SUFFIX = {
    ArtifactType.IMAGE: ".png",
    ArtifactType.MASK: ".png",
    ArtifactType.DEPTH: ".png",
    ArtifactType.TEXT: ".txt",
    ArtifactType.JSON: ".json",
    ArtifactType.BOUNDING_BOXES: ".json",
    ArtifactType.POSE: ".json",
    ArtifactType.SCENE_GRAPH: ".json",
}


class ArtifactStore:
    """Persist payloads once globally and expose run-scoped immutable references."""

    def __init__(self, layout: StorageLayout, run_id: str) -> None:
        self.layout = layout.ensure()
        self.run_id = run_id
        self.layout.ensure_run(run_id)

    @property
    def result_manifests_dir(self) -> Path:
        path = self.layout.artifacts_dir(self.run_id) / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _serialize(payload: ArtifactPayload) -> bytes:
        if payload.data is not None:
            return payload.data
        if payload.text is not None:
            return payload.text.encode("utf-8")
        if payload.json_value is not None:
            return canonical_json_bytes(payload.json_value)
        assert payload.source_path is not None
        source = Path(payload.source_path)
        if not source.is_file():
            raise FileNotFoundError(f"artifact source is not a file: {source}")
        return source.read_bytes()

    def put(self, payload: ArtifactPayload) -> ArtifactRef:
        data = self._serialize(payload)
        digest = sha256_bytes(data)
        suffix = payload.suffix or _DEFAULT_SUFFIX.get(payload.artifact_type, "")
        artifact_id = f"artifact_{payload.artifact_type.value}_{digest[:20]}"
        object_path = self.layout.object_path(digest, suffix=suffix)
        if not object_path.exists():
            try:
                atomic_write_bytes(object_path, data, overwrite=False)
            except FileExistsError:
                pass
        target = self.layout.artifact_path(self.run_id, artifact_id, suffix=suffix)
        if not target.exists():
            try:
                atomic_copy(object_path, target, overwrite=False)
            except FileExistsError:
                pass
        return ArtifactRef(
            artifact_type=payload.artifact_type,
            artifact_id=artifact_id,
            uri=self.layout.relative_uri(target),
            mime_type=payload.mime_type,
            shape=payload.shape,
            dtype=payload.dtype,
            sha256=digest,
            frame_id=payload.frame_id,
            metadata={
                "object_uri": self.layout.relative_uri(object_path),
                **payload.metadata,
            },
        )

    def put_many(
        self, payloads: tuple[ArtifactPayload, ...]
    ) -> tuple[ArtifactRef, ...]:
        refs: list[ArtifactRef] = []
        counts: dict[str, int] = {}
        for payload in payloads:
            ref = self.put(payload)
            occurrence = counts.get(ref.artifact_id, 0)
            counts[ref.artifact_id] = occurrence + 1
            if occurrence:
                ref = replace(ref, artifact_id=f"{ref.artifact_id}_{occurrence + 1}")
            refs.append(ref)
        return tuple(refs)

    def resolve(self, artifact: ArtifactRef | str) -> Path:
        uri = artifact.uri if isinstance(artifact, ArtifactRef) else artifact
        return self.layout.resolve_uri(uri)

    def record_result(self, result: ToolResult) -> Path:
        path = self.result_manifests_dir / f"{result.result_id}.json"
        atomic_write_json(path, result.to_dict())
        return path

    def read_result(self, result_id: str) -> ToolResult:
        path = self.result_manifests_dir / f"{result_id}.json"
        return ToolResult.from_dict(read_json(path))

    def iter_results(self) -> tuple[ToolResult, ...]:
        return tuple(
            ToolResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.result_manifests_dir.glob("*.json"))
        )


__all__ = ["ArtifactStore"]
