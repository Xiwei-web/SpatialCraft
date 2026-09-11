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
    bundle_scalar,
    camera_intrinsics,
    map_pixels,
    pixel_mapping,
    point_set,
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
        version="2.1.0",
        description="Compute 2D/3D geometry. Keep coordinate_length_unit separate from value_unit; frame_id aliases result_frame_id. Explicit length_unit never proves metric scale: pass scale_status or preserve artifact provenance. points_uri accepts 3D point sets or valid projected pixels. Camera operations use camera_to_world and intrinsics; pixel_space/intrinsics_space default processed, and source_to_processed is required if they differ. Projection emits a distinct pixel frame. For backprojection, source_frame_id labels pixels; target_frame_id (legacy frame_id fallback) labels world output; camera-z depths use length_unit.",
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
                "source_frame_id": {"type": "string", "minLength": 1},
                "scale_status": {"type": "string", "minLength": 1},
                "pixel_space": enum_schema("processed", "source"),
                "intrinsics_space": enum_schema("processed", "source"),
                "source_to_processed": array_schema(
                    array_schema({"type": "number"}, min_items=3, max_items=3),
                    min_items=3,
                    max_items=3,
                ),
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
        import hashlib
        import json

        import numpy as np

        operation = arguments["operation"]
        backprojection = operation == "backproject_points"
        camera_operation = operation in {"project_points", "backproject_points"}
        dimension = 2 if backprojection else 3
        bundle = {}
        points = None
        if "points_uri" in arguments:
            if "points" in arguments:
                raise ToolSchemaError("provide points or points_uri, not both")
            points, bundle = point_set(str(arguments["points_uri"]), dimension)

        def bound_scalar(argument, key, default):
            stored = bundle_scalar(bundle, key)
            if stored and argument in arguments and str(arguments[argument]) != stored:
                raise ToolSchemaError(
                    f"points artifact {key} differs from declared {argument}"
                )
            return str(arguments.get(argument, stored or default))

        unit = bound_scalar(
            "length_unit",
            "world_length_unit" if backprojection else "length_unit",
            "reconstruction_unit",
        )
        if unit not in {"meter", "reconstruction_unit"}:
            raise ToolSchemaError(
                "3D point/depth length_unit must be meter or reconstruction_unit"
            )
        scale_status = bound_scalar("scale_status", "scale_status", "unverified")
        if backprojection:
            world_frame = str(
                arguments.get(
                    "target_frame_id",
                    arguments.get(
                        "frame_id",
                        bundle_scalar(bundle, "world_frame_id", "geometry:world"),
                    ),
                )
            )
            stored_world = bundle_scalar(bundle, "world_frame_id")
            if stored_world and world_frame != stored_world:
                raise ToolSchemaError(
                    "projection artifact camera_to_world belongs to a different world frame"
                )
            source_frame = bound_scalar("source_frame_id", "frame_id", "")
            result_frame = world_frame
        else:
            default_frame = arguments.get(
                "frame_id",
                arguments.get(
                    "first_frame_id", arguments.get("second_frame_id", "geometry:world")
                ),
            )
            source_frame = bound_scalar("source_frame_id", "frame_id", default_frame)
            if "frame_id" in arguments and str(arguments["frame_id"]) != source_frame:
                raise ToolSchemaError(
                    "points artifact/source_frame_id differs from declared frame_id"
                )
            world_frame = source_frame
            result_frame = str(arguments.get("target_frame_id", source_frame))
        for key in ("first_frame_id", "second_frame_id"):
            if key in arguments and str(arguments[key]) != source_frame:
                raise ToolSchemaError(
                    "operands must share a coordinate frame; transform explicitly first"
                )

        camera_metadata = {}
        if camera_operation:
            calibration = {}
            for key in ("intrinsics", "camera_to_world", "source_to_processed"):
                value = arguments.get(key, bundle.get(key))
                if value is None:
                    if key == "source_to_processed":
                        continue
                    raise ToolSchemaError(
                        "camera operation requires intrinsics and camera_to_world"
                    )
                if backprojection and key in bundle and key in arguments:
                    declared, stored = np.asarray(value), bundle[key]
                    if declared.shape != stored.shape or not np.allclose(
                        declared, stored
                    ):
                        raise ToolSchemaError(
                            f"projection artifact {key} differs from declared calibration"
                        )
                calibration[key] = value
            intrinsics = camera_intrinsics(calibration["intrinsics"])
            c2w = se3(calibration["camera_to_world"])
            pixel_space = bound_scalar("pixel_space", "pixel_space", "processed")
            intrinsics_space = bound_scalar(
                "intrinsics_space", "intrinsics_space", "processed"
            )
            mapping = calibration.get("source_to_processed")
            if mapping is not None:
                mapping = pixel_mapping(mapping)
            if pixel_space != intrinsics_space and mapping is None:
                raise ToolSchemaError(
                    "different pixel_space/intrinsics_space require source_to_processed"
                )
            camera_metadata = {
                "pixel_space": pixel_space,
                "intrinsics_space": intrinsics_space,
                "intrinsics": intrinsics.tolist(),
                "camera_to_world": c2w.tolist(),
                "world_frame_id": world_frame,
                "world_length_unit": unit,
                "pixel_convention": "integer coordinates are pixel centers; x=right,y=down",
            }
            if mapping is not None:
                camera_metadata["source_to_processed"] = mapping.tolist()
            identity = hashlib.sha256(
                json.dumps(camera_metadata, sort_keys=True).encode()
            ).hexdigest()[:16]
            pixel_frame = f"geometry:pixel:{identity}:{pixel_space}"
            if backprojection:
                source_frame = source_frame or pixel_frame
            else:
                result_frame = str(arguments.get("target_frame_id", pixel_frame))
            if source_frame == result_frame:
                raise ToolSchemaError(
                    "pixel and world coordinates require distinct frame IDs"
                )

        source_unit = "pixel" if backprojection else unit
        coordinate_unit = "pixel" if operation == "project_points" else unit
        output: dict[str, Any] = {
            "operation": operation,
            "source_frame_id": source_frame,
            "result_frame_id": result_frame,
            "frame_id": result_frame,
            "source_coordinate_length_unit": source_unit,
            "coordinate_length_unit": coordinate_unit,
            "value_unit": coordinate_unit,
            "scale_status": scale_status,
        }
        artifacts = ()
        if operation in {
            "point_distance",
            "angle_between_vectors",
            "rotation_matrix_from_vectors",
        }:
            if result_frame != source_frame:
                raise ToolSchemaError(
                    "a scalar/vector result cannot relabel its source coordinate frame"
                )
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
                        value_unit="degree",
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
                    output.update(
                        rotation=rotation.tolist(), value_unit="dimensionless"
                    )
        else:
            if points is None:
                points = np.asarray(arguments.get("points", ()), dtype=float)
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
                valid = np.ones(len(result), dtype=bool)
            elif operation == "project_points":
                camera = transform_points(points, np.linalg.inv(c2w))
                valid = camera[:, 2] > 0
                result = np.full((len(camera), 2), np.nan)
                projected = camera[valid] @ intrinsics.T
                pixels = projected[:, :2] / projected[:, 2:3]
                if pixel_space != intrinsics_space:
                    pixels = map_pixels(
                        pixels, mapping, inverse=pixel_space == "source"
                    )
                result[valid] = pixels
                output["validity"] = "positive camera-z; image bounds are not checked"
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
                if pixel_space != intrinsics_space:
                    points = map_pixels(
                        points, mapping, inverse=intrinsics_space == "source"
                    )
                rays = (
                    np.column_stack((points, np.ones(len(points))))
                    @ np.linalg.inv(intrinsics).T
                )
                result = transform_points(rays * depth[:, None], c2w)
                valid = np.ones(len(result), dtype=bool)
            limit = int(arguments.get("max_output_points", 16))
            output.update(
                points=[
                    point.tolist() if ok else None
                    for point, ok in zip(result[:limit], valid[:limit], strict=True)
                ],
                valid=valid[:limit].tolist(),
                point_count=len(result),
                points_truncated=len(result) > limit,
            )
            output.update(camera_metadata)
            payload = {
                "points": result,
                "valid": valid,
                "frame_id": result_frame,
                "length_unit": coordinate_unit,
                "scale_status": scale_status,
                "source_frame_id": source_frame,
                "source_coordinate_length_unit": source_unit,
                "coordinate_length_unit": coordinate_unit,
                "value_unit": output["value_unit"],
            }
            if "source_image_uri" in bundle:
                payload["source_image_uri"] = bundle_scalar(bundle, "source_image_uri")
            if operation == "project_points":
                payload.update(pixels=result, **camera_metadata)
            artifacts = (
                ArtifactPayload(
                    artifact_type=ArtifactType.JSON
                    if operation == "project_points"
                    else ArtifactType.POINT_CLOUD,
                    data=array_bundle(**payload),
                    suffix=".npz",
                    mime_type="application/x-npz",
                    shape=result.shape,
                    dtype="float64",
                    frame_id=result_frame,
                    metadata={
                        "content": "projected_pixels"
                        if operation == "project_points"
                        else "points",
                        "length_unit": coordinate_unit,
                        "scale_status": scale_status,
                        "source_frame_id": source_frame,
                        "result_frame_id": result_frame,
                        "coordinate_length_unit": coordinate_unit,
                        "value_unit": output["value_unit"],
                        "downstream": ["geometry.backproject_points"]
                        if operation == "project_points"
                        else ["geometry.transform_points", "geometry.project_points"],
                        **camera_metadata,
                    },
                ),
            )
        output["unit"] = output["value_unit"]  # legacy result-value alias
        frames = []
        for identifier, length in (
            (source_frame, source_unit),
            (result_frame, coordinate_unit),
        ):
            if any(frame.frame_id == identifier for frame in frames):
                continue
            frames.append(
                CoordinateFrame(
                    frame_id=identifier,
                    unit=length,
                    convention=(
                        "integer pixel centers; x=right,y=down"
                        if length == "pixel"
                        else "explicit input coordinate frame"
                    ),
                    metadata=camera_metadata
                    if length == "pixel"
                    else {"scale_status": scale_status},
                )
            )
        return ToolExecution(
            text=f"Geometry {operation}: {output}",
            structured_output=output,
            artifacts=artifacts,
            coordinate_frames=tuple(frames),
            unit=output["value_unit"],
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
        frame_id = str(
            arguments.get(
                "source_frame_id", arguments.get("frame_id", "geometry:frame")
            )
        )
        if "frame_id" in arguments and str(arguments["frame_id"]) != frame_id:
            raise ToolSchemaError("source_frame_id differs from declared frame_id")
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
        coordinate_unit = str(output.get("unit", unit))
        value_unit = (
            "dimensionless" if operation == "bbox_relation" else coordinate_unit
        )
        result_frame = str(
            arguments.get(
                "target_frame_id",
                (
                    f"{frame_id}:{coordinate_unit}"
                    if coordinate_unit != unit
                    else frame_id
                ),
            )
        )
        if result_frame == frame_id and coordinate_unit != unit:
            raise ToolSchemaError(
                "different coordinate units require distinct frame IDs"
            )
        frames = (frame,)
        if result_frame != frame_id:
            frames += (
                CoordinateFrame(
                    frame_id=result_frame,
                    unit=coordinate_unit,
                    convention="x-right,y-down",
                ),
            )
        output.update(
            frame_id=result_frame,
            source_frame_id=frame_id,
            result_frame_id=result_frame,
            source_coordinate_length_unit=unit,
            coordinate_length_unit=coordinate_unit,
            value_unit=value_unit,
            unit=value_unit,
        )
        return ToolExecution(
            text=text,
            structured_output=output,
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.JSON,
                    json_value=output,
                    suffix=".json",
                    mime_type="application/json",
                    frame_id=result_frame,
                ),
            ),
            coordinate_frames=frames,
            unit=value_unit,
        )


__all__ = ["GeometryTool"]
