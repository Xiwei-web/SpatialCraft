"""Coordinate-frame constructors, validation, and image-space conversion."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from spatialcraft.schemas import CoordinateFrame


def image_pixel_frame(image_id: str, width: int, height: int) -> CoordinateFrame:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    return CoordinateFrame(
        frame_id=f"{image_id}:pixel",
        unit="pixel",
        convention="origin=top-left,x=right,y=down; boxes=[x1,y1,x2,y2]",
        metadata={"width": width, "height": height},
    )


def image_normalized_frame(image_id: str, width: int, height: int) -> CoordinateFrame:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    return CoordinateFrame(
        frame_id=f"{image_id}:normalized",
        parent_frame_id=f"{image_id}:pixel",
        transform_to_parent=(
            float(width - 1),
            0.0,
            0.0,
            0.0,
            0.0,
            float(height - 1),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        unit="normalized",
        convention="origin=top-left,x=right,y=down,range=[0,1]",
        metadata={"width": width, "height": height},
    )


def validate_bbox(
    bbox: Sequence[float], *, width: int | None = None, height: int | None = None
) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("a bounding box requires [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 < x1 or y2 < y1:
        raise ValueError("bounding box maxima must not precede minima")
    if width is not None and not 0 <= x1 <= x2 <= width - 1:
        raise ValueError("bounding box x coordinates exceed image bounds")
    if height is not None and not 0 <= y1 <= y2 <= height - 1:
        raise ValueError("bounding box y coordinates exceed image bounds")
    return x1, y1, x2, y2


def convert_points(
    points: Iterable[Sequence[float]],
    *,
    width: int,
    height: int,
    source_unit: str,
    target_unit: str,
) -> tuple[tuple[float, float], ...]:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if source_unit not in {"pixel", "normalized"} or target_unit not in {
        "pixel",
        "normalized",
    }:
        raise ValueError("source_unit and target_unit must be pixel or normalized")
    converted: list[tuple[float, float]] = []
    for point in points:
        if len(point) != 2:
            raise ValueError("each point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if source_unit == target_unit:
            converted.append((x, y))
        elif source_unit == "normalized":
            converted.append((x * (width - 1), y * (height - 1)))
        else:
            converted.append((x / max(1, width - 1), y / max(1, height - 1)))
    return tuple(converted)


__all__ = [
    "convert_points",
    "image_normalized_frame",
    "image_pixel_frame",
    "validate_bbox",
]
