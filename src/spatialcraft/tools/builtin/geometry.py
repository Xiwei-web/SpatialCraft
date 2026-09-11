"""Dependency-free geometric calculations for tool-composition tests and agents."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..coordinate_frames import convert_points, validate_bbox
from ..schema_builder import ToolSchemaError, array_schema, enum_schema, object_schema
from ..spatial_arrays import (
    array_bundle,
    camera_intrinsics,
    read_bundle,
    se3,
    transform_points,
)


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
        version="2.0.0",
        description="Compute 2D box relations/conversion, 2D or 3D point distance, 3D angles/rotations, SE(3) transforms and camera projection/backprojection. Use explicit length_unit and frame_id for 3D coordinates; project/backproject use pixel intrinsics and camera_to_world. points_uri accepts masked_points artifacts.",
        input_schema=object_schema(
            {
                "operation": enum_schema(
                    "bbox_relation",
                    "point_distance",
                    "convert_points",
                    "angle_between_vectors",
                    "transform_points",
                    "project_points",
                    "backproject_points",
                    "rotation_matrix_from_vectors",
                ),
                "first": array_schema({"type": "number"}, min_items=2, max_items=4),
                "second": array_schema({"type": "number"}, min_items=2, max_items=4),
                "points": array_schema(
                    array_schema({"type": "number"}, min_items=2, max_items=3)
                ),
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
                "source_unit": enum_schema("pixel", "normalized"),
                "target_unit": enum_schema("pixel", "normalized"),
                "frame_id": {"type": "string", "minLength": 1},
                "target_frame_id": {"type": "string", "minLength": 1},
                "first_frame_id": {"type": "string", "minLength": 1},
                "second_frame_id": {"type": "string", "minLength": 1},
                "length_unit": enum_schema("meter", "reconstruction_unit"),
                "matrix": array_schema(
                    array_schema({"type": "number"}, min_items=4, max_items=4),
                    min_items=4,
                    max_items=4,
                ),
                "camera_to_world": array_schema(
                    array_schema({"type": "number"}, min_items=4, max_items=4),
                    min_items=4,
                    max_items=4,
                ),
                "intrinsics": array_schema(
                    array_schema({"type": "number"}, min_items=3, max_items=3),
                    min_items=3,
                    max_items=3,
                ),
                "depth_values": array_schema({"type": "number", "exclusiveMinimum": 0}),
                "points_uri": {"type": "string", "minLength": 1},
                "max_output_points": {"type": "integer", "minimum": 0, "maximum": 128},
            },
            required=("operation",),
        ),
    )

    def _three_dimensional(self, arguments: Mapping[str, Any]) -> ToolExecution:
        import numpy as np

        operation = arguments["operation"]
        unit = str(arguments.get("length_unit", "reconstruction_unit"))
        frame_id = str(arguments.get("frame_id", "geometry:world"))
        target_frame_id = str(arguments.get("target_frame_id", frame_id))
        artifacts = ()
        output: dict[str, Any] = {
            "operation": operation,
            "frame_id": frame_id,
            "unit": unit,
        }
        if arguments.get("first_frame_id", frame_id) != arguments.get(
            "second_frame_id", frame_id
        ):
            raise ToolSchemaError(
                "operands must share a coordinate frame; transform explicitly first"
            )
        if operation in {
            "point_distance",
            "angle_between_vectors",
            "rotation_matrix_from_vectors",
        }:
            first, second = (
                np.asarray(arguments.get(key, ()), dtype=float)
                for key in ("first", "second")
            )
            if (
                first.shape != (3,)
                or second.shape != (3,)
                or not np.isfinite([first, second]).all()
            ):
                raise ToolSchemaError("operation requires two finite 3D vectors")
            if operation == "point_distance":
                output["distance"] = float(np.linalg.norm(first - second))
            else:
                if min(np.linalg.norm(first), np.linalg.norm(second)) <= 1e-12:
                    raise ToolSchemaError(
                        "angle/rotation is undefined for a zero vector"
                    )
                first, second = (
                    first / np.linalg.norm(first),
                    second / np.linalg.norm(second),
                )
                cosine = float(np.clip(np.dot(first, second), -1, 1))
                if operation == "angle_between_vectors":
                    output.update(
                        angle_degrees=float(np.degrees(np.arccos(cosine))),
                        unit="degree",
                    )
                else:
                    axis = np.cross(first, second)
                    sine = np.linalg.norm(axis)
                    if sine < 1e-10:
                        rotation = np.eye(3)
                        if cosine < 0:
                            basis = np.eye(3)[np.argmin(np.abs(first))]
                            axis = np.cross(first, basis)
                            axis /= np.linalg.norm(axis)
                            rotation = 2 * np.outer(axis, axis) - np.eye(3)
                    else:
                        x, y, z = axis
                        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                        rotation = (
                            np.eye(3) + cross + cross @ cross * ((1 - cosine) / sine**2)
                        )
                    output.update(rotation=rotation.tolist(), unit="dimensionless")
        else:
            if "points_uri" in arguments:
                bundle = read_bundle(str(arguments["points_uri"]))
                if "points" not in bundle or bundle["points"].ndim != 2:
                    raise ToolSchemaError(
                        "points_uri must be a masked/geometry point-set artifact"
                    )
                points = np.asarray(bundle["points"], dtype=float)
                stored_frame = (
                    str(bundle.get("frame_id", "").item())
                    if "frame_id" in bundle
                    else ""
                )
                stored_unit = (
                    str(bundle.get("length_unit", "").item())
                    if "length_unit" in bundle
                    else ""
                )
                if (
                    "frame_id" in arguments
                    and stored_frame
                    and frame_id != stored_frame
                ):
                    raise ToolSchemaError(
                        "points artifact frame differs from declared frame"
                    )
                if "length_unit" in arguments and stored_unit and unit != stored_unit:
                    raise ToolSchemaError(
                        "points artifact unit differs from declared unit"
                    )
                frame_id, unit = stored_frame or frame_id, stored_unit or unit
                target_frame_id = str(arguments.get("target_frame_id", frame_id))
                output.update(frame_id=frame_id, unit=unit)
            else:
                points = np.asarray(arguments.get("points", ()), dtype=float)
            dimension = 2 if operation == "backproject_points" else 3
            if (
                points.ndim != 2
                or points.shape[1] != dimension
                or not np.isfinite(points).all()
            ):
                raise ToolSchemaError(f"operation requires finite Nx{dimension} points")
            if operation == "transform_points":
                if "matrix" not in arguments:
                    raise ToolSchemaError("transform_points requires matrix")
                result = transform_points(points, arguments["matrix"])
                output["frame_id"] = target_frame_id
            else:
                if "intrinsics" not in arguments or "camera_to_world" not in arguments:
                    raise ToolSchemaError(
                        "camera operation requires intrinsics and camera_to_world"
                    )
                intrinsics = camera_intrinsics(arguments["intrinsics"])
                c2w = se3(arguments["camera_to_world"])
                if operation == "project_points":
                    camera = transform_points(points, np.linalg.inv(c2w))
                    valid = camera[:, 2] > 0
                    result = np.full((len(camera), 2), np.nan)
                    projected = camera[valid] @ intrinsics.T
                    result[valid] = projected[:, :2] / projected[:, 2:3]
                    limit = int(arguments.get("max_output_points", 16))
                    output.update(
                        points=[
                            p.tolist() if ok else None
                            for p, ok in zip(result[:limit], valid[:limit], strict=True)
                        ],
                        valid=valid[:limit].tolist(),
                        point_count=len(result),
                        points_truncated=len(result) > limit,
                        unit="pixel",
                        pixel_convention="x=right,y=down",
                    )
                    artifacts = (
                        ArtifactPayload(
                            artifact_type=ArtifactType.JSON,
                            data=array_bundle(pixels=result, valid=valid),
                            suffix=".npz",
                            mime_type="application/x-npz",
                            shape=result.shape,
                            dtype="float64",
                            frame_id=frame_id,
                            metadata={
                                "content": "projected_pixels",
                                "validity": "positive camera-z",
                            },
                        ),
                    )
                else:
                    depth = np.asarray(arguments.get("depth_values", ()), dtype=float)
                    if (
                        depth.shape != (len(points),)
                        or not np.isfinite(depth).all()
                        or (depth <= 0).any()
                    ):
                        raise ToolSchemaError(
                            "backproject_points requires one finite positive camera-z depth per pixel"
                        )
                    rays = (
                        np.column_stack((points, np.ones(len(points))))
                        @ np.linalg.inv(intrinsics).T
                    )
                    result = transform_points(rays * depth[:, None], c2w)
                    output["frame_id"] = target_frame_id
            if operation != "project_points":
                limit = int(arguments.get("max_output_points", 16))
                output.update(
                    points=result[:limit].tolist(),
                    point_count=len(result),
                    points_truncated=len(result) > limit,
                    unit=unit,
                )
                artifacts = (
                    ArtifactPayload(
                        artifact_type=ArtifactType.POINT_CLOUD,
                        data=array_bundle(
                            points=result, frame_id=output["frame_id"], length_unit=unit
                        ),
                        suffix=".npz",
                        mime_type="application/x-npz",
                        shape=result.shape,
                        dtype="float64",
                        frame_id=output["frame_id"],
                        metadata={
                            "content": "points",
                            "length_unit": unit,
                            "downstream": ["geometry.transform_points"],
                        },
                    ),
                )
        frame = CoordinateFrame(
            frame_id=output["frame_id"],
            unit=output["unit"],
            convention="explicit input coordinate frame",
        )
        return ToolExecution(
            text=f"Geometry {operation}: {output}",
            structured_output=output,
            artifacts=artifacts,
            coordinate_frames=(frame,),
            unit=output["unit"],
        )

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        operation = arguments["operation"]
        if operation in {
            "angle_between_vectors",
            "transform_points",
            "project_points",
            "backproject_points",
            "rotation_matrix_from_vectors",
        }:
            return self._three_dimensional(arguments)
        if operation == "point_distance" and len(arguments.get("first", ())) == 3:
            return self._three_dimensional(arguments)
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
