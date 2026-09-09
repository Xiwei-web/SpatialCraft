"""Depth Anything 3 multi-view reconstruction adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..schema_builder import array_schema, object_schema
from .common import (
    LazyResource,
    SpatialToolPaths,
    add_python_path,
    normalized_preview,
    numpy_bytes,
)

ReconstructionPredictor = Callable[[Sequence[str]], Mapping[str, Any]]


class DepthAnything3Adapter:
    def __init__(
        self,
        *,
        paths: SpatialToolPaths | None = None,
        device: str = "cuda",
        predictor: ReconstructionPredictor | None = None,
    ) -> None:
        self.paths = paths or SpatialToolPaths.from_env()
        self.device = device
        self.predictor = predictor
        self._runtime = LazyResource(self._load)

    @property
    def loaded(self) -> bool:
        return self.predictor is not None or self._runtime.loaded

    def _load(self):
        add_python_path(self.paths.da3_repo / "src")
        if not self.paths.da3_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Depth Anything 3 checkpoint not found: {self.paths.da3_checkpoint}"
            )
        from depth_anything_3.api import DepthAnything3

        return DepthAnything3.from_pretrained(str(self.paths.da3_checkpoint)).to(
            device=self.device
        )

    def predict(self, image_uris: Sequence[str]) -> Mapping[str, Any]:
        if self.predictor is not None:
            return self.predictor(image_uris)
        prediction = self._runtime.get().inference(list(image_uris))
        return {
            name: getattr(prediction, name, None)
            for name in (
                "processed_images",
                "depth",
                "conf",
                "extrinsics",
                "intrinsics",
            )
        }


class ReconstructionTool(SpatialTool):
    spec = ToolSpec(
        name="reconstruct",
        description="Reconstruct consistent multi-view depth and camera geometry with Depth Anything 3.",
        input_schema=object_schema(
            {
                "image_uris": array_schema(
                    {"type": "string", "minLength": 1}, min_items=1, max_items=64
                )
            },
            required=("image_uris",),
        ),
        metadata={"model": "Depth Anything 3"},
        default_timeout_s=300,
    )

    def __init__(self, adapter: DepthAnything3Adapter | None = None) -> None:
        self.adapter = adapter or DepthAnything3Adapter()

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        import numpy as np

        image_uris = tuple(str(item) for item in arguments["image_uris"])
        prediction = self.adapter.predict(image_uris)
        depth = np.asarray(prediction["depth"], dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[None]
        confidence = np.asarray(
            prediction.get("conf")
            if prediction.get("conf") is not None
            else np.ones_like(depth),
            dtype=np.float32,
        )
        intrinsics = np.asarray(prediction.get("intrinsics"), dtype=np.float32)
        extrinsics = np.asarray(prediction.get("extrinsics"), dtype=np.float32)
        frames = []
        for index in range(len(depth)):
            transform = None
            if extrinsics.size and index < len(extrinsics):
                matrix = extrinsics[index]
                if matrix.shape == (3, 4):
                    matrix = np.vstack([matrix, [0, 0, 0, 1]])
                if matrix.shape == (4, 4):
                    camera_to_world = np.linalg.inv(matrix)
                    transform = tuple(
                        float(value) for value in camera_to_world.reshape(-1)
                    )
            frames.append(
                CoordinateFrame(
                    frame_id=f"reconstruction:camera:{index}",
                    parent_frame_id="reconstruction:world" if transform else None,
                    transform_to_parent=transform,
                    unit="meter",
                    convention="OpenCV camera pose in reconstruction world",
                    metadata={"source_extrinsics": "world_to_camera"},
                )
            )
        world = CoordinateFrame(
            frame_id="reconstruction:world",
            unit="meter",
            convention="Depth Anything 3 reconstruction world",
        )
        frames.append(world)
        bundle = numpy_bytes(
            depth,
            compressed=True,
            confidence=confidence,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
        artifacts = [
            ArtifactPayload(
                artifact_type=ArtifactType.POINT_CLOUD,
                data=bundle,
                suffix=".npz",
                mime_type="application/x-npz",
                shape=depth.shape,
                dtype=str(depth.dtype),
                frame_id=world.frame_id,
                metadata={"content": "depth-confidence-intrinsics-extrinsics"},
            )
        ]
        for index, item in enumerate(depth):
            artifacts.append(
                ArtifactPayload(
                    artifact_type=ArtifactType.IMAGE,
                    data=normalized_preview(item),
                    suffix=".png",
                    mime_type="image/png",
                    shape=item.shape,
                    dtype="uint8",
                    frame_id=frames[index].frame_id,
                    metadata={"role": "depth_preview", "view_index": index},
                )
            )
        mean_confidence = float(np.clip(np.nanmean(confidence), 0, 1))
        return ToolExecution(
            text=f"Depth Anything 3 reconstructed {len(depth)} view(s).",
            structured_output={
                "view_count": len(depth),
                "depth_shape": list(depth.shape),
                "mean_confidence": mean_confidence,
                "camera_frame_ids": [frame.frame_id for frame in frames[:-1]],
                "world_frame_id": world.frame_id,
            },
            artifacts=tuple(artifacts),
            coordinate_frames=tuple(frames),
            confidence=mean_confidence,
            unit="meter",
            metadata={"backend_model": "Depth Anything 3"},
        )


__all__ = ["DepthAnything3Adapter", "ReconstructionTool"]
