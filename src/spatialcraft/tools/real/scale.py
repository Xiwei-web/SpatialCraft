"""MoGe-2 metric geometry and scale adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..coordinate_frames import image_pixel_frame
from ..schema_builder import object_schema
from .common import (
    LazyResource,
    SpatialToolPaths,
    add_python_path,
    normalized_preview,
    numpy_bytes,
    require_file,
)

ScalePredictor = Callable[[str, int], Mapping[str, Any]]


class MoGe2Adapter:
    def __init__(
        self,
        *,
        paths: SpatialToolPaths | None = None,
        device: str = "cuda",
        predictor: ScalePredictor | None = None,
    ) -> None:
        self.paths = paths or SpatialToolPaths.from_env()
        self.device = device
        self.predictor = predictor
        self._runtime = LazyResource(self._load)

    @property
    def loaded(self) -> bool:
        return self.predictor is not None or self._runtime.loaded

    def _load(self):
        add_python_path(self.paths.moge_repo)
        checkpoint = require_file(self.paths.moge_checkpoint, "MoGe-2 checkpoint")
        from moge.model.v2 import MoGeModel

        return MoGeModel.from_pretrained(str(checkpoint)).to(self.device).eval()

    def predict(self, image_uri: str, resolution_level: int) -> Mapping[str, Any]:
        if self.predictor is not None:
            return self.predictor(image_uri, resolution_level)
        import cv2
        import torch

        image = cv2.cvtColor(cv2.imread(image_uri), cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(f"cannot read image: {image_uri}")
        tensor = torch.tensor(
            image / 255.0, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)
        output = self._runtime.get().infer(tensor, resolution_level=resolution_level)
        return {
            key: value.detach().float().cpu().numpy()
            if hasattr(value, "detach")
            else value
            for key, value in output.items()
        }


class ScaleTool(SpatialTool):
    spec = ToolSpec(
        name="scale",
        description="Estimate metric depth, camera intrinsics, and point geometry with MoGe-2.",
        input_schema=object_schema(
            {
                "image_uri": {"type": "string", "minLength": 1},
                "resolution_level": {"type": "integer", "minimum": 0, "maximum": 9},
            },
            required=("image_uri",),
        ),
        metadata={"model": "MoGe-2"},
    )

    def __init__(self, adapter: MoGe2Adapter | None = None) -> None:
        self.adapter = adapter or MoGe2Adapter()

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        import numpy as np

        prediction = self.adapter.predict(
            str(arguments["image_uri"]), int(arguments.get("resolution_level", 6))
        )
        points = np.asarray(prediction["points"], dtype=np.float32)
        depth = np.asarray(prediction.get("depth", points[..., 2]), dtype=np.float32)
        intrinsics = np.asarray(
            prediction.get("intrinsics", np.eye(3)), dtype=np.float32
        )
        height, width = depth.shape[-2:]
        pixel_frame = image_pixel_frame(
            Path(arguments["image_uri"]).stem, width, height
        )
        camera_frame = CoordinateFrame(
            frame_id=f"{Path(arguments['image_uri']).stem}:camera",
            unit="meter",
            convention="OpenCV camera: x=right,y=down,z=forward",
        )
        finite = depth[np.isfinite(depth)]
        output = {
            "min_depth_m": float(finite.min()),
            "max_depth_m": float(finite.max()),
            "median_depth_m": float(np.median(finite)),
            "intrinsics": intrinsics.tolist(),
            "width": width,
            "height": height,
            "pixel_frame_id": pixel_frame.frame_id,
            "camera_frame_id": camera_frame.frame_id,
        }
        return ToolExecution(
            text=(
                f"MoGe-2 estimated metric depth median "
                f"{output['median_depth_m']:.3f} m."
            ),
            structured_output=output,
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.DEPTH,
                    data=numpy_bytes(depth),
                    suffix=".npy",
                    mime_type="application/x-npy",
                    shape=depth.shape,
                    dtype=str(depth.dtype),
                    frame_id=pixel_frame.frame_id,
                    metadata={"value_unit": "meter"},
                ),
                ArtifactPayload(
                    artifact_type=ArtifactType.IMAGE,
                    data=normalized_preview(depth),
                    suffix=".png",
                    mime_type="image/png",
                    shape=(height, width),
                    dtype="uint8",
                    frame_id=pixel_frame.frame_id,
                    metadata={"role": "depth_preview"},
                ),
                ArtifactPayload(
                    artifact_type=ArtifactType.POINT_CLOUD,
                    data=numpy_bytes(points, compressed=True, intrinsics=intrinsics),
                    suffix=".npz",
                    mime_type="application/x-npz",
                    shape=points.shape,
                    dtype=str(points.dtype),
                    frame_id=camera_frame.frame_id,
                ),
            ),
            coordinate_frames=(pixel_frame, camera_frame),
            confidence=0.9,
            unit="meter",
            metadata={"backend_model": "MoGe-2"},
        )


__all__ = ["MoGe2Adapter", "ScaleTool"]
