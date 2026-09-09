"""Tolerance-aware numeric answer verification."""

from __future__ import annotations

import re
from typing import Any

from spatialcraft.schemas import TaskSample, VerifierOutcome

from .base import AnswerVerifier, make_outcome
from .exact_match import unwrap_answer

_NUMBER = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?\s*%?"
)


def parse_number(value: Any) -> float:
    text = unwrap_answer(str(value))
    match = _NUMBER.search(text)
    if match is None:
        raise ValueError(f"no numeric value found in {text!r}")
    token = match.group(0).replace(",", "").replace(" ", "")
    percentage = token.endswith("%")
    token = token.removesuffix("%")
    if "/" in token:
        numerator, denominator = token.split("/", maxsplit=1)
        value_float = float(numerator) / float(denominator)
    else:
        value_float = float(token)
    return value_float / 100.0 if percentage else value_float


class NumericVerifier(AnswerVerifier):
    name = "numeric"

    def __init__(
        self, *, absolute_tolerance: float = 1e-3, relative_tolerance: float = 1e-2
    ) -> None:
        if absolute_tolerance < 0 or relative_tolerance < 0:
            raise ValueError("numeric tolerances cannot be negative")
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)

    def verify_answer(self, task: TaskSample, candidate: str) -> VerifierOutcome:
        try:
            predicted = parse_number(candidate)
            expected = parse_number(task.reference_answer)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            return make_outcome(
                self.name,
                0.0,
                is_correct=False,
                analysis=str(exc),
                details={"candidate": candidate, "reason": "invalid_number"},
            )
        absolute_tolerance = float(
            task.metadata.get("numeric_absolute_tolerance", self.absolute_tolerance)
        )
        relative_tolerance = float(
            task.metadata.get("numeric_relative_tolerance", self.relative_tolerance)
        )
        absolute_error = abs(predicted - expected)
        allowed_error = max(absolute_tolerance, relative_tolerance * abs(expected))
        correct = absolute_error <= allowed_error
        return make_outcome(
            self.name,
            float(correct),
            is_correct=correct,
            details={
                "candidate": candidate,
                "predicted_value": predicted,
                "expected_value": expected,
                "absolute_error": absolute_error,
                "allowed_error": allowed_error,
            },
        )


__all__ = ["NumericVerifier", "parse_number"]
