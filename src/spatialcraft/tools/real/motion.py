"""Dense Farneback optical-flow motion tool."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..coordinate_frames import image_pixel_frame
from ..schema_builder import object_schema
from .common import numpy_bytes, png_bytes


class FarnebackMotionTool(SpatialTool):
    spec = ToolSpec(
        name="motion",
        version="2.0.0",
        description="Estimate dense pixel displacement between two frames with Farneback flow.",
        input_schema=object_schema(
            {
                "first_image_uri": {"type": "string", "minLength": 1},
                "second_image_uri": {"type": "string", "minLength": 1},
                "mask_uri": {"type": "string", "minLength": 1},
                "pyramid_scale": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
                "levels": {"type": "integer", "minimum": 1, "maximum": 10},
                "window_size": {"type": "integer", "minimum": 3, "maximum": 101},
            },
            required=("first_image_uri", "second_image_uri"),
        ),
        metadata={"algorithm": "OpenCV Farneback"},
    )

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        import cv2
        import numpy as np

        first = cv2.imread(str(arguments["first_image_uri"]), cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(str(arguments["second_image_uri"]), cv2.IMREAD_GRAYSCALE)
        if first is None or second is None:
            raise FileNotFoundError("motion input image could not be read")
        if first.shape != second.shape:
            raise ValueError("motion frames must have identical dimensions")
        flow = cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            float(arguments.get("pyramid_scale", 0.5)),
            int(arguments.get("levels", 3)),
            int(arguments.get("window_size", 15)),
            3,
            5,
            1.2,
            0,
        )
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        valid = np.isfinite(flow).all(axis=-1)
        if "mask_uri" in arguments:
            mask = cv2.imread(str(arguments["mask_uri"]), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != first.shape:
                raise ValueError(
                    "motion mask must exist and match the input frame dimensions"
                )
            valid &= mask > 0
        samples = flow[valid]
        magnitudes = magnitude[valid]
        hsv = np.zeros((*first.shape, 3), dtype=np.uint8)
        hsv[..., 0] = angle * 90 / np.pi
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        height, width = first.shape
        frame = image_pixel_frame(
            Path(arguments["first_image_uri"]).stem, width, height
        )
        output = {
            "available": bool(len(samples)),
            "valid_pixel_count": len(samples),
            "mean_dx": float(samples[:, 0].mean()) if len(samples) else None,
            "mean_dy": float(samples[:, 1].mean()) if len(samples) else None,
            "median_flow": np.median(samples, axis=0).tolist()
            if len(samples)
            else None,
            "mean_magnitude": float(magnitudes.mean()) if len(samples) else None,
            "max_magnitude": float(magnitudes.max()) if len(samples) else None,
            "interpretation": "image-plane displacement in pixels; not metric camera motion",
            "width": width,
            "height": height,
            "frame_id": frame.frame_id,
        }
        return ToolExecution(
            text=(
                f"Mean motion dx={output['mean_dx']}, "
                f"dy={output['mean_dy']} pixels; {len(samples)} valid pixels."
            ),
            structured_output=output,
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.MASK,
                    data=png_bytes(valid.astype(np.uint8) * 255, mode="L"),
                    suffix=".png",
                    mime_type="image/png",
                    shape=valid.shape,
                    dtype="uint8",
                    frame_id=frame.frame_id,
                    metadata={
                        "role": "flow_validity",
                        "source_image_uri": str(arguments["first_image_uri"]),
                    },
                ),
                ArtifactPayload(
                    artifact_type=ArtifactType.OPTICAL_FLOW,
                    data=numpy_bytes(flow),
                    suffix=".npy",
                    mime_type="application/x-npy",
                    shape=flow.shape,
                    dtype=str(flow.dtype),
                    frame_id=frame.frame_id,
                    metadata={"component_order": ["dx", "dy"]},
                ),
                ArtifactPayload(
                    artifact_type=ArtifactType.IMAGE,
                    data=png_bytes(visualization),
                    suffix=".png",
                    mime_type="image/png",
                    shape=visualization.shape,
                    dtype="uint8",
                    frame_id=frame.frame_id,
                    metadata={"role": "flow_visualization"},
                ),
            ),
            coordinate_frames=(frame,),
            confidence=1.0,
            unit="pixel",
        )


__all__ = ["FarnebackMotionTool"]
