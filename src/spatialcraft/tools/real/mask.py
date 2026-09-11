"""Mask algebra and morphology tool."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..coordinate_frames import image_pixel_frame
from ..schema_builder import array_schema, enum_schema, object_schema
from ..spatial_arrays import array_bundle, masked_world_points
from .common import png_bytes


class MaskTool(SpatialTool):
    spec = ToolSpec(
        name="mask",
        version="2.0.0",
        description="Combine masks, compute 2D statistics, or extract centroid_3d/masked_points from a reconstruction_uri. 3D extraction uses exactly one source-image mask and absolute frame_index; output is in the reconstruction frame and unverified units unless calibrated.",
        input_schema=object_schema(
            {
                "mask_uris": array_schema(
                    {"type": "string", "minLength": 1}, min_items=1
                ),
                "operation": enum_schema(
                    "union",
                    "intersection",
                    "difference",
                    "invert",
                    "dilate",
                    "erode",
                    "statistics",
                    "iou",
                    "centroid_3d",
                    "masked_points",
                ),
                "reconstruction_uri": {"type": "string", "minLength": 1},
                "frame_index": {"type": "integer", "minimum": 0},
                "source_image_uri": {"type": "string", "minLength": 1},
                "mask_space": enum_schema("source", "processed"),
                "min_confidence": {"type": "number"},
                "min_points": {"type": "integer", "minimum": 1},
                "kernel_size": {"type": "integer", "minimum": 1, "maximum": 101},
                "iterations": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            required=("mask_uris", "operation"),
        ),
    )

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        if arguments["operation"] in {"centroid_3d", "masked_points"}:
            return self._extract_3d(arguments)
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
        iou = None
        if operation == "iou":
            if len(masks) != 2:
                raise ValueError("mask IoU requires exactly two masks")
            output = masks[0] & masks[1]
            union_count = int((masks[0] | masks[1]).sum())
            iou = float(output.sum() / union_count) if union_count else 1.0
        elif operation == "statistics":
            output = masks[0]
        elif operation == "union":
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
        ys, xs = np.nonzero(output)
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            if foreground
            else None
        )
        centroid = [float(np.median(xs)), float(np.median(ys))] if foreground else None
        robust_bbox = (
            [
                float(np.quantile(xs, 0.01)),
                float(np.quantile(ys, 0.01)),
                float(np.quantile(xs, 0.99)),
                float(np.quantile(ys, 0.99)),
            ]
            if foreground > 100
            else bbox
        )
        return ToolExecution(
            text=f"Mask {operation} produced {foreground} foreground pixels.",
            structured_output={
                "operation": operation,
                "bbox_xyxy": bbox,
                "robust_bbox_xyxy": robust_bbox,
                "iou": iou,
                "centroid_semantics": "coordinate-wise median of foreground pixels",
                "mask_statistics": [
                    {
                        "area": int(mask.sum()),
                        "centroid_xy": [
                            float(np.median(np.nonzero(mask)[1])),
                            float(np.median(np.nonzero(mask)[0])),
                        ]
                        if mask.any()
                        else None,
                    }
                    for mask in masks
                ]
                if operation == "statistics"
                else None,
                "centroid_xy": centroid,
                "available": bool(foreground),
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

    def _extract_3d(self, arguments: Mapping[str, Any]) -> ToolExecution:
        import numpy as np

        from ..schema_builder import ToolSchemaError

        if "reconstruction_uri" not in arguments:
            raise ToolSchemaError("3D mask extraction requires reconstruction_uri")
        points, view = masked_world_points(arguments)
        available = len(points) >= int(arguments.get("min_points", 3))
        frame = CoordinateFrame(
            frame_id=view["world_frame_id"],
            unit=view["length_unit"],
            convention="source reconstruction world; not implicitly gravity aligned",
        )
        output = {
            "operation": arguments["operation"],
            "available": available,
            "point_count": len(points),
            "centroid": np.median(points, axis=0).tolist() if available else None,
            "centroid_semantics": "coordinate-wise median of visible valid masked surface points",
            "frame_id": frame.frame_id,
            "frame_index": view["frame_index"],
            "length_unit": view["length_unit"],
            "scale_status": view["scale_status"],
            "source_image_uri": str(view["source_image_uris"]),
            "reason": None if available else "insufficient_valid_masked_points",
        }
        artifacts = ()
        if available:
            artifacts = (
                ArtifactPayload(
                    artifact_type=ArtifactType.POINT_CLOUD,
                    data=array_bundle(
                        points=points,
                        frame_id=frame.frame_id,
                        length_unit=view["length_unit"],
                        scale_status=view["scale_status"],
                        source_image_uri=str(view["source_image_uris"]),
                    ),
                    suffix=".npz",
                    mime_type="application/x-npz",
                    shape=points.shape,
                    dtype="float64",
                    frame_id=frame.frame_id,
                    metadata={
                        "content": "masked_points",
                        "length_unit": view["length_unit"],
                        "downstream": ["geometry.transform_points"],
                        "source_reconstruction_uri": str(
                            arguments["reconstruction_uri"]
                        ),
                    },
                ),
            )
        return ToolExecution(
            text=f"3D mask extraction: {output}",
            structured_output=output,
            artifacts=artifacts,
            coordinate_frames=(frame,),
            unit=view["length_unit"],
        )


__all__ = ["MaskTool"]
