"""Basic annotation renderer used by agents and mock/real tool adapters."""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spatialcraft.schemas import ArtifactType

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..coordinate_frames import image_pixel_frame, validate_bbox
from ..schema_builder import array_schema, object_schema

_BOX_SCHEMA = object_schema(
    {
        "bbox": array_schema({"type": "number"}, min_items=4, max_items=4),
        "label": {"type": "string"},
        "color": {"type": "string", "default": "red"},
    },
    required=("bbox",),
)
_POINT_SCHEMA = object_schema(
    {
        "point": array_schema({"type": "number"}, min_items=2, max_items=2),
        "label": {"type": "string"},
        "color": {"type": "string", "default": "lime"},
    },
    required=("point",),
)


def render_annotations(
    image_uri: str,
    *,
    boxes: list[Mapping[str, Any]] | None = None,
    points: list[Mapping[str, Any]] | None = None,
) -> tuple[bytes, int, int]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required by the draw tool") from exc
    path = Path(image_uri)
    if not path.is_file():
        raise FileNotFoundError(f"input image not found: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for item in boxes or []:
        bbox = validate_bbox(item["bbox"], width=width, height=height)
        color = str(item.get("color", "red"))
        draw.rectangle(bbox, outline=color, width=max(1, min(width, height) // 200))
        if item.get("label"):
            draw.text((bbox[0] + 2, bbox[1] + 2), str(item["label"]), fill=color)
    radius = max(2, min(width, height) // 100)
    for item in points or []:
        x, y = (float(value) for value in item["point"])
        color = str(item.get("color", "lime"))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if item.get("label"):
            draw.text((x + radius, y), str(item["label"]), fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), width, height


class DrawTool(SpatialTool):
    spec = ToolSpec(
        name="draw",
        description="Draw labeled boxes and points on an image and return the annotation.",
        input_schema=object_schema(
            {
                "image_uri": {"type": "string", "minLength": 1},
                "boxes": array_schema(_BOX_SCHEMA),
                "points": array_schema(_POINT_SCHEMA),
            },
            required=("image_uri",),
        ),
    )

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        data, width, height = render_annotations(
            arguments["image_uri"],
            boxes=arguments.get("boxes"),
            points=arguments.get("points"),
        )
        frame = image_pixel_frame(Path(arguments["image_uri"]).stem, width, height)
        return ToolExecution(
            text=(
                f"Rendered {len(arguments.get('boxes', []))} box(es) and "
                f"{len(arguments.get('points', []))} point(s)."
            ),
            structured_output={
                "width": width,
                "height": height,
                "box_count": len(arguments.get("boxes", [])),
                "point_count": len(arguments.get("points", [])),
            },
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.IMAGE,
                    data=data,
                    suffix=".png",
                    mime_type="image/png",
                    shape=(height, width, 3),
                    dtype="uint8",
                    frame_id=frame.frame_id,
                    metadata={"role": "annotation"},
                ),
            ),
            coordinate_frames=(frame,),
            unit="pixel",
        )


__all__ = ["DrawTool", "render_annotations"]
