"""Multiple-choice answer resolution and verification."""

from __future__ import annotations

import re
from typing import Any

from spatialcraft.schemas import TaskSample, VerifierOutcome

from .base import AnswerVerifier, make_outcome
from .exact_match import normalize_answer, unwrap_answer

_EXPLICIT_LABEL = re.compile(
    r"(?:final\s+answer|answer|option|choice)\s*(?:is|:|：)?\s*"
    r"[\[(]?([A-Z]|\d{1,2})[\])]?(?:\s|[.,;:!?]|$)",
    re.IGNORECASE,
)
_LEADING_LABEL = re.compile(r"^\s*[\[(]?([A-Z]|\d{1,2})[\])]?(?:\s*[.)：:]|\s*$)")


def _label_to_index(token: str, choice_count: int) -> int | None:
    if token.isdigit():
        value = int(token)
        index = value - 1 if value > 0 else 0
    else:
        index = ord(token.upper()) - ord("A")
    return index if 0 <= index < choice_count else None


def _normalize_choice_text(value: Any) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", normalize_answer(value))


def resolve_choice(value: Any, choices: tuple[str, ...]) -> int | None:
    """Resolve a label, 1-based number, or exact choice text to an index."""

    text = unwrap_answer(str(value)).strip()
    normalized = _normalize_choice_text(text)
    exact = [
        index
        for index, choice in enumerate(choices)
        if _normalize_choice_text(choice) == normalized
    ]
    if len(exact) == 1:
        return exact[0]
    for pattern in (_LEADING_LABEL, _EXPLICIT_LABEL):
        match = pattern.search(text)
        if match:
            return _label_to_index(match.group(1), len(choices))
    return None


class MultipleChoiceVerifier(AnswerVerifier):
    name = "multiple_choice"

    def verify_answer(self, task: TaskSample, candidate: str) -> VerifierOutcome:
        labels = tuple(task.metadata.get("choice_labels") or ())
        reference = task.reference_answer
        expected_index: int | None = None
        if labels:
            normalized_reference = normalize_answer(reference).upper()
            normalized_labels = tuple(str(label).strip().upper() for label in labels)
            if normalized_reference in normalized_labels:
                expected_index = normalized_labels.index(normalized_reference)
        if expected_index is None:
            expected_index = resolve_choice(reference, task.choices)
        predicted_index = resolve_choice(candidate, task.choices)
        correct = expected_index is not None and predicted_index == expected_index
        return make_outcome(
            self.name,
            float(correct),
            is_correct=correct,
            details={
                "candidate": candidate,
                "predicted_index": predicted_index,
                "predicted_label": (
                    chr(ord("A") + predicted_index)
                    if predicted_index is not None and predicted_index < 26
                    else None
                ),
                "expected_index": expected_index,
                "expected_label": (
                    chr(ord("A") + expected_index)
                    if expected_index is not None and expected_index < 26
                    else None
                ),
            },
        )


__all__ = ["MultipleChoiceVerifier", "resolve_choice"]
