"""Deterministic mock monocular depth estimator."""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..coordinate_frames import image_pixel_frame
from ..schema_builder import object_schema


class MockDepthTool(SpatialTool):
    spec = ToolSpec(
        name="depth",
        description="Estimate a metric depth map for one image.",
        input_schema=object_schema(
            {
                "image_uri": {"type": "string", "minLength": 1},
                "near_m": {"type": "number", "exclusiveMinimum": 0},
                "far_m": {"type": "number", "exclusiveMinimum": 0},
            },
            required=("image_uri",),
        ),
        metadata={"implementation": "mock"},
    )

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("NumPy and Pillow are required by mock depth") from exc
        image_uri = str(arguments["image_uri"])
        near = float(arguments.get("near_m", 0.5))
        far = float(arguments.get("far_m", 5.0))
        if far <= near:
            raise ValueError("far_m must be greater than near_m")
        with Image.open(image_uri) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        height, width = gray.shape
        vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        depth = near + (far - near) * (0.75 * vertical + 0.25 * (1.0 - gray))
        millimeters = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        preview = (
            (depth - depth.min()) / max(1e-6, depth.max() - depth.min()) * 255
        ).astype(np.uint8)
        depth_buffer, preview_buffer = io.BytesIO(), io.BytesIO()
        Image.fromarray(millimeters, mode="I;16").save(depth_buffer, format="PNG")
        Image.fromarray(preview, mode="L").save(preview_buffer, format="PNG")
        frame = image_pixel_frame(Path(image_uri).stem, width, height)
        output = {
            "min_depth_m": float(depth.min()),
            "max_depth_m": float(depth.max()),
            "mean_depth_m": float(depth.mean()),
            "width": width,
            "height": height,
            "frame_id": frame.frame_id,
        }
        return ToolExecution(
            text=(
                f"Estimated depth from {output['min_depth_m']:.3f} to "
                f"{output['max_depth_m']:.3f} meters."
            ),
            structured_output=output,
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.DEPTH,
                    data=depth_buffer.getvalue(),
                    suffix=".png",
                    mime_type="image/png",
                    shape=(height, width),
                    dtype="uint16",
                    frame_id=frame.frame_id,
                    metadata={"value_unit": "millimeter", "scale_to_meter": 0.001},
                ),
                ArtifactPayload(
                    artifact_type=ArtifactType.IMAGE,
                    data=preview_buffer.getvalue(),
                    suffix=".png",
                    mime_type="image/png",
                    shape=(height, width),
                    dtype="uint8",
                    frame_id=frame.frame_id,
                    metadata={"role": "depth_preview"},
                ),
            ),
            coordinate_frames=(frame,),
            confidence=0.75,
            unit="meter",
            metadata={"mock": True},
        )


__all__ = ["MockDepthTool"]
