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
        description="Estimate dense pixel displacement between two frames with Farneback flow.",
        input_schema=object_schema(
            {
                "first_image_uri": {"type": "string", "minLength": 1},
                "second_image_uri": {"type": "string", "minLength": 1},
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
            "mean_dx": float(flow[..., 0].mean()),
            "mean_dy": float(flow[..., 1].mean()),
            "mean_magnitude": float(magnitude.mean()),
            "max_magnitude": float(magnitude.max()),
            "width": width,
            "height": height,
            "frame_id": frame.frame_id,
        }
        return ToolExecution(
            text=(
                f"Mean motion dx={output['mean_dx']:.3f}, "
                f"dy={output['mean_dy']:.3f} pixels."
            ),
            structured_output=output,
            artifacts=(
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
