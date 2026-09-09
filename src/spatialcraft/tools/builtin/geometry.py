"""Dependency-free geometric calculations for tool-composition tests and agents."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..coordinate_frames import convert_points, validate_bbox
from ..schema_builder import ToolSchemaError, array_schema, enum_schema, object_schema


def _bbox_relation(first: tuple[float, ...], second: tuple[float, ...]) -> list[str]:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    relations = ["left" if acx < bcx else "right" if acx > bcx else "aligned-x"]
    relations.append("above" if acy < bcy else "below" if acy > bcy else "aligned-y")
    if ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2:
        relations.append("contains")
    elif bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2:
        relations.append("inside")
    elif max(ax1, bx1) <= min(ax2, bx2) and max(ay1, by1) <= min(ay2, by2):
        relations.append("overlap")
    else:
        relations.append("disjoint")
    return relations


class GeometryTool(SpatialTool):
    spec = ToolSpec(
        name="geometry",
        description="Compute 2D box relations, point distance, or pixel/normalized conversion. For bbox_relation, first/second must be [x_min,y_min,x_max,y_max] with maxima >= minima; for point_distance, each is [x,y].",
        input_schema=object_schema(
            {
                "operation": enum_schema(
                    "bbox_relation", "point_distance", "convert_points"
                ),
                "first": array_schema({"type": "number"}, min_items=2, max_items=4),
                "second": array_schema({"type": "number"}, min_items=2, max_items=4),
                "points": array_schema(
                    array_schema({"type": "number"}, min_items=2, max_items=2)
                ),
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
                "source_unit": enum_schema("pixel", "normalized"),
                "target_unit": enum_schema("pixel", "normalized"),
                "frame_id": {"type": "string", "minLength": 1},
            },
            required=("operation",),
        ),
    )

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        operation = arguments["operation"]
        frame_id = str(arguments.get("frame_id", "geometry:frame"))
        unit = str(arguments.get("source_unit", "pixel"))
        frame = CoordinateFrame(
            frame_id=frame_id, unit=unit, convention="x-right,y-down"
        )
        if operation == "bbox_relation":
            try:
                first = validate_bbox(arguments.get("first", ()))
                second = validate_bbox(arguments.get("second", ()))
            except ValueError as exc:
                raise ToolSchemaError(str(exc)) from exc
            output: dict[str, Any] = {"relations": _bbox_relation(first, second)}
            text = "Relations: " + ", ".join(output["relations"])
        elif operation == "point_distance":
            first = tuple(float(value) for value in arguments.get("first", ()))
            second = tuple(float(value) for value in arguments.get("second", ()))
            if len(first) != 2 or len(second) != 2:
                raise ToolSchemaError("point_distance requires two 2D points")
            output = {"distance": math.dist(first, second), "unit": unit}
            text = f"Distance: {output['distance']:.6g} {unit}"
        else:
            required = ("points", "width", "height", "source_unit", "target_unit")
            missing = [key for key in required if key not in arguments]
            if missing:
                raise ToolSchemaError(f"convert_points missing arguments: {missing}")
            try:
                converted = convert_points(
                    arguments["points"],
                    width=int(arguments["width"]),
                    height=int(arguments["height"]),
                    source_unit=arguments["source_unit"],
                    target_unit=arguments["target_unit"],
                )
            except ValueError as exc:
                raise ToolSchemaError(str(exc)) from exc
            output = {
                "points": [list(point) for point in converted],
                "unit": arguments["target_unit"],
            }
            text = f"Converted {len(converted)} point(s) to {arguments['target_unit']}"
        return ToolExecution(
            text=text,
            structured_output=output,
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.JSON,
                    json_value=output,
                    suffix=".json",
                    mime_type="application/json",
                    frame_id=frame_id,
                ),
            ),
            coordinate_frames=(frame,),
            unit=str(output.get("unit", unit)),
        )


__all__ = ["GeometryTool"]
