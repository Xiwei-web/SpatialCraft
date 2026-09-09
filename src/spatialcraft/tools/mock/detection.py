"""Deterministic mock detector that exercises the production tool contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..builtin.draw import render_annotations
from ..coordinate_frames import image_pixel_frame
from ..schema_builder import array_schema, object_schema


class MockDetectionTool(SpatialTool):
    spec = ToolSpec(
        name="detect",
        description="Locate queried objects and return pixel-space boxes with confidence.",
        input_schema=object_schema(
            {
                "image_uri": {"type": "string", "minLength": 1},
                "queries": array_schema(
                    {"type": "string", "minLength": 1}, min_items=1, max_items=32
                ),
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "max_detections": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            required=("image_uri", "queries"),
        ),
        metadata={"implementation": "mock"},
    )

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required by mock detection") from exc
        image_uri = str(arguments["image_uri"])
        with Image.open(image_uri) as image:
            width, height = image.size
        threshold = float(arguments.get("threshold", 0.25))
        maximum = int(arguments.get("max_detections", len(arguments["queries"])))
        detections: list[dict[str, Any]] = []
        for query in list(arguments["queries"])[:maximum]:
            digest = hashlib.sha256(
                f"{Path(image_uri).name}\0{query}".encode()
            ).digest()
            confidence = 0.5 + digest[0] / 510
            if confidence < threshold:
                continue
            box_width = max(2, int(width * (0.15 + digest[1] / 1024)))
            box_height = max(2, int(height * (0.15 + digest[2] / 1024)))
            x1 = int((digest[3] / 255) * max(0, width - box_width - 1))
            y1 = int((digest[4] / 255) * max(0, height - box_height - 1))
            detections.append(
                {
                    "label": str(query),
                    "bbox": [x1, y1, x1 + box_width, y1 + box_height],
                    "confidence": round(confidence, 6),
                }
            )
        frame = image_pixel_frame(Path(image_uri).stem, width, height)
        output = {
            "detections": detections,
            "width": width,
            "height": height,
            "frame_id": frame.frame_id,
        }
        annotated, _, _ = render_annotations(image_uri, boxes=detections)
        mean_confidence = (
            sum(item["confidence"] for item in detections) / len(detections)
            if detections
            else 0.0
        )
        return ToolExecution(
            text=f"Detected {len(detections)} object instance(s).",
            structured_output=output,
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.BOUNDING_BOXES,
                    text=json.dumps(output, ensure_ascii=False),
                    suffix=".json",
                    mime_type="application/json",
                    frame_id=frame.frame_id,
                ),
                ArtifactPayload(
                    artifact_type=ArtifactType.IMAGE,
                    data=annotated,
                    suffix=".png",
                    mime_type="image/png",
                    shape=(height, width, 3),
                    dtype="uint8",
                    frame_id=frame.frame_id,
                    metadata={"role": "detection_overlay"},
                ),
            ),
            coordinate_frames=(frame,),
            confidence=mean_confidence,
            unit="pixel",
            metadata={"mock": True},
        )


__all__ = ["MockDetectionTool"]
