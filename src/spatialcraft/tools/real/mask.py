"""Mask algebra and morphology tool."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..coordinate_frames import image_pixel_frame
from ..schema_builder import array_schema, enum_schema, object_schema
from .common import png_bytes


class MaskTool(SpatialTool):
    spec = ToolSpec(
        name="mask",
        description="Combine, invert, dilate, or erode persisted binary masks.",
        input_schema=object_schema(
            {
                "mask_uris": array_schema(
                    {"type": "string", "minLength": 1}, min_items=1
                ),
                "operation": enum_schema(
                    "union", "intersection", "difference", "invert", "dilate", "erode"
                ),
                "kernel_size": {"type": "integer", "minimum": 1, "maximum": 101},
                "iterations": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            required=("mask_uris", "operation"),
        ),
    )

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        import cv2
        import numpy as np
        from PIL import Image

        masks = []
        for uri in arguments["mask_uris"]:
            with Image.open(uri) as image:
                masks.append(np.asarray(image.convert("L")) > 0)
        if any(mask.shape != masks[0].shape for mask in masks):
            raise ValueError("all masks must share the same shape")
        operation = arguments["operation"]
        if operation == "union":
            output = np.logical_or.reduce(masks)
        elif operation == "intersection":
            output = np.logical_and.reduce(masks)
        elif operation == "difference":
            output = masks[0].copy()
            for mask in masks[1:]:
                output &= ~mask
        elif operation == "invert":
            output = ~masks[0]
        else:
            kernel_size = int(arguments.get("kernel_size", 3))
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            source = masks[0].astype(np.uint8)
            function = cv2.dilate if operation == "dilate" else cv2.erode
            output = function(
                source, kernel, iterations=int(arguments.get("iterations", 1))
            ).astype(bool)
        height, width = output.shape
        frame = image_pixel_frame(Path(arguments["mask_uris"][0]).stem, width, height)
        foreground = int(output.sum())
        return ToolExecution(
            text=f"Mask {operation} produced {foreground} foreground pixels.",
            structured_output={
                "operation": operation,
                "foreground_pixels": foreground,
                "foreground_fraction": foreground / output.size,
                "width": width,
                "height": height,
                "frame_id": frame.frame_id,
            },
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.MASK,
                    data=png_bytes(output.astype(np.uint8) * 255, mode="L"),
                    suffix=".png",
                    mime_type="image/png",
                    shape=(height, width),
                    dtype="uint8",
                    frame_id=frame.frame_id,
                ),
            ),
            coordinate_frames=(frame,),
            confidence=1.0,
            unit="pixel",
        )


__all__ = ["MaskTool"]
