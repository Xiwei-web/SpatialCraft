"""Portable array contracts shared by reconstruction, mask, pose and scale.

NPZ files contain their coordinate/provenance metadata; downstream consumers do
not infer a world frame or metric scale from a filename. Arrays never use pickle.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from .schema_builder import ToolSchemaError


def array_bundle(**values: Any) -> bytes:
    import numpy as np

    stream = io.BytesIO()
    np.savez_compressed(
        stream, **{key: np.asarray(value) for key, value in values.items()}
    )
    return stream.getvalue()


def read_bundle(uri: str) -> dict[str, Any]:
    import numpy as np

    with np.load(uri, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def se3(value: Any) -> Any:
    import numpy as np

    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, [0, 0, 0, 1]))
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ToolSchemaError("transform must be a finite 4x4 SE(3) matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3)
    ):
        raise ToolSchemaError(
            "transform must have a proper rotation and homogeneous last row"
        )
    return matrix


def transform_points(points: Any, matrix: Any) -> Any:
    import numpy as np

    values = np.asarray(points, dtype=float)
    if values.shape[-1:] != (3,) or not np.isfinite(values).all():
        raise ToolSchemaError("points must be finite 3D coordinates")
    transform = se3(matrix)
    return values @ transform[:3, :3].T + transform[:3, 3]


def camera_intrinsics(value: Any) -> Any:
    import numpy as np

    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
    ):
        raise ToolSchemaError(
            "intrinsics must be a finite pixel-space 3x3 camera matrix"
        )
    return matrix


def backproject(depth: Any, intrinsics: Any, camera_to_world: Any) -> Any:
    import numpy as np

    depth = np.asarray(depth, dtype=float)
    yy, xx = np.indices(depth.shape)
    rays = (
        np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        @ np.linalg.inv(camera_intrinsics(intrinsics)).T
    )
    camera = rays * depth[..., None]
    transform = se3(camera_to_world)
    return camera @ transform[:3, :3].T + transform[:3, 3]


def reconstruction_view(uri: str, frame_index: int = 0) -> dict[str, Any]:
    import numpy as np

    bundle = read_bundle(uri)
    required = {
        "points",
        "depth",
        "valid",
        "intrinsics",
        "camera_to_world",
        "frame_indices",
        "world_frame_id",
        "length_unit",
        "scale_status",
        "source_image_uris",
        "source_shapes",
        "source_to_processed",
    }
    missing = required - bundle.keys()
    if missing:
        raise ToolSchemaError(
            f"reconstruction artifact lacks v2 contract fields: {sorted(missing)}"
        )
    matches = np.flatnonzero(bundle["frame_indices"] == frame_index)
    if len(matches) != 1:
        raise ToolSchemaError(
            f"frame_index {frame_index} does not uniquely identify a reconstruction view"
        )
    position = int(matches[0])
    result = {
        key: bundle[key][position]
        for key in (
            "points",
            "depth",
            "valid",
            "intrinsics",
            "camera_to_world",
            "source_shapes",
            "source_to_processed",
            "source_image_uris",
        )
    }
    for key in ("world_frame_id", "length_unit", "scale_status"):
        result[key] = str(bundle[key].item())
    result["frame_index"] = frame_index
    if "confidence" in bundle:
        result["confidence"] = bundle["confidence"][position]
    return result


def align_source_array(
    array: Any, view: dict[str, Any], *, nearest: bool = True
) -> Any:
    """Apply the recorded source-to-processed map, with no silent shape guess."""
    import cv2
    import numpy as np

    values = np.asarray(array)
    if tuple(values.shape[:2]) != tuple(view["source_shapes"]):
        raise ToolSchemaError(
            "array dimensions do not match the reconstruction source image"
        )
    height, width = view["depth"].shape
    matrix = np.asarray(view["source_to_processed"], dtype=float)
    # cv2.resize uses pixel-center resampling, matching SAM masks and DA3 resize.
    sx, sy = width / values.shape[1], height / values.shape[0]
    expected = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    if np.allclose(matrix, expected):
        return cv2.resize(
            values,
            (width, height),
            interpolation=cv2.INTER_NEAREST_EXACT if nearest else cv2.INTER_LINEAR,
        )
    return cv2.warpPerspective(
        values,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def masked_world_points(arguments: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from PIL import Image

    view = reconstruction_view(
        str(arguments["reconstruction_uri"]), int(arguments.get("frame_index", 0))
    )
    source = arguments.get("source_image_uri")
    if (
        source is not None
        and Path(source).resolve() != Path(str(view["source_image_uris"])).resolve()
    ):
        raise ToolSchemaError(
            "mask source image does not match the reconstruction view"
        )
    uri = arguments.get("mask_uri")
    if uri is None:
        masks = arguments.get("mask_uris", ())
        if len(masks) != 1:
            raise ToolSchemaError("3D extraction requires exactly one mask")
        uri = masks[0]
    with Image.open(uri) as image:
        mask = np.asarray(image.convert("L")) > 0
    if arguments.get("mask_space", "source") == "processed":
        if mask.shape != view["depth"].shape:
            raise ToolSchemaError(
                "processed mask dimensions do not match reconstruction"
            )
    else:
        mask = align_source_array(mask.astype("uint8"), view).astype(bool)
    points = np.asarray(view["points"], dtype=float)
    valid = (
        mask & np.asarray(view["valid"], dtype=bool) & np.isfinite(points).all(axis=-1)
    )
    if "min_confidence" in arguments:
        if "confidence" not in view:
            raise ToolSchemaError(
                "confidence threshold requested but artifact has no confidence"
            )
        valid &= np.asarray(view["confidence"]) >= float(arguments["min_confidence"])
    return points[valid], view
