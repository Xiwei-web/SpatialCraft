"""Spatial-relation and point-in-mask verification."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

from spatialcraft.schemas import AnswerType, TaskSample, VerifierOutcome

from .base import AnswerVerifier, make_outcome
from .exact_match import normalize_answer, unwrap_answer

_ALIASES = {
    "left of": "left",
    "to the left of": "left",
    "right of": "right",
    "to the right of": "right",
    "on top of": "above",
    "over": "above",
    "under": "below",
    "beneath": "below",
    "in front of": "front",
    "front of": "front",
    "at the back of": "behind",
    "far from": "far",
    "close to": "near",
    "next to": "near",
    "within": "inside",
    "contained in": "inside",
    "surrounding": "contains",
    "intersecting": "overlap",
    "overlapping": "overlap",
    "separate": "disjoint",
}


def canonical_relation(value: Any) -> str:
    text = normalize_answer(value).replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text in _ALIASES:
        return _ALIASES[text]
    compounds = {
        "front left": "front-left",
        "left front": "front-left",
        "front right": "front-right",
        "right front": "front-right",
        "back left": "back-left",
        "left back": "back-left",
        "back right": "back-right",
        "right back": "back-right",
        "upper left": "upper-left",
        "upper right": "upper-right",
        "lower left": "lower-left",
        "lower right": "lower-right",
    }
    return compounds.get(text, text)


def _relation_set(value: Any) -> frozenset[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(canonical_relation(item) for item in value)
    text = unwrap_answer(str(value))
    parts = re.split(r"\s*(?:,|;|/|\band\b)\s*", text, flags=re.IGNORECASE)
    return frozenset(canonical_relation(part) for part in parts if part.strip())


def parse_points(value: Any) -> tuple[tuple[float, float], ...]:
    raw = value
    if isinstance(value, str):
        text = unwrap_answer(value)
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            try:
                raw = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError("point answer must be a list of [x, y] pairs") from exc
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 2
        and all(isinstance(item, (int, float)) for item in raw)
    ):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("point answer must contain at least one point")
    points: list[tuple[float, float]] = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each point must contain exactly x and y")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("point coordinates must be finite")
        points.append((x, y))
    return tuple(points)


def _score_against_mask(
    points: tuple[tuple[float, float], ...], mask_uri: str
) -> float:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for point-in-mask verification") from exc
    path = Path(mask_uri)
    if not path.is_file():
        raise FileNotFoundError(f"mask image not found: {path}")
    with Image.open(path) as image:
        mask = image.convert("L")
        width, height = mask.size
        hits = 0
        for x, y in points:
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                continue
            column = min(width - 1, round(x * (width - 1)))
            row = min(height - 1, round(y * (height - 1)))
            hits += int(mask.getpixel((column, row)) > 0)
    return hits / len(points)


def _score_against_reference(
    predicted: tuple[tuple[float, float], ...],
    expected: tuple[tuple[float, float], ...],
    tolerance: float,
) -> float:
    hits = sum(
        any(math.dist(point, target) <= tolerance for target in expected)
        for point in predicted
    )
    return hits / max(len(predicted), len(expected))


class SpatialRelationVerifier(AnswerVerifier):
    name = "spatial_relation"

    def verify_answer(self, task: TaskSample, candidate: str) -> VerifierOutcome:
        if task.answer_type is AnswerType.POINTING:
            return self._verify_points(task, candidate)
        predicted = _relation_set(candidate)
        expected = _relation_set(task.reference_answer)
        correct = bool(predicted) and predicted == expected
        return make_outcome(
            self.name,
            float(correct),
            is_correct=correct,
            details={
                "candidate": candidate,
                "predicted_relations": sorted(predicted),
                "expected_relations": sorted(expected),
            },
        )

    def _verify_points(self, task: TaskSample, candidate: str) -> VerifierOutcome:
        try:
            predicted = parse_points(candidate)
            mask_uri = task.metadata.get("mask_uri")
            if mask_uri:
                score = _score_against_mask(predicted, str(mask_uri))
                method = "point_in_mask"
            else:
                expected = parse_points(task.reference_answer)
                tolerance = float(task.metadata.get("point_tolerance", 0.05))
                score = _score_against_reference(predicted, expected, tolerance)
                method = "reference_distance"
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            return make_outcome(
                self.name,
                0.0,
                is_correct=False,
                analysis=str(exc),
                details={"candidate": candidate, "reason": "invalid_point_answer"},
            )
        threshold = float(task.metadata.get("pointing_pass_threshold", 0.5))
        return make_outcome(
            self.name,
            score,
            is_correct=score >= threshold,
            details={
                "candidate": candidate,
                "predicted_points": predicted,
                "method": method,
                "pass_threshold": threshold,
            },
        )


__all__ = ["SpatialRelationVerifier", "canonical_relation", "parse_points"]
