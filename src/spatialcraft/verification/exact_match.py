"""Unicode-aware exact-match verification."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from spatialcraft.schemas import AnswerType, TaskSample, VerifierOutcome

from .base import AnswerVerifier, make_outcome

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_PREFIX = re.compile(
    r"^\s*(?:final\s+answer|answer|response)\s*[:：]\s*",
    re.IGNORECASE,
)
_TERMINAL_PUNCTUATION = " .,!?:;。！？，：；"


def unwrap_answer(text: str) -> str:
    """Remove common final-answer wrappers without touching inner content."""

    match = _ANSWER_TAG.search(text)
    if match:
        text = match.group(1)
    else:
        finals = re.findall(
            r"^\s*Final\s+Answer\s*[:：]\s*(.+?)\s*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if finals:
            text = finals[-1]
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            lines = lines[1:-1]
            text = "\n".join(lines)
    return _PREFIX.sub("", text).strip()


def normalize_answer(value: Any) -> str:
    text = unwrap_answer(str(value))
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text).strip(_TERMINAL_PUNCTUATION)
    boolean = {
        "true": "yes",
        "false": "no",
        "y": "yes",
        "n": "no",
    }
    return boolean.get(text, text)


def _structured_equal(reference: Any, candidate: str) -> bool | None:
    if not isinstance(reference, (Mapping, list, tuple)):
        return None
    try:
        parsed = json.loads(unwrap_answer(candidate))
    except (TypeError, json.JSONDecodeError):
        return False
    expected = list(reference) if isinstance(reference, tuple) else reference
    return parsed == expected


class ExactMatchVerifier(AnswerVerifier):
    """Binary exact match with optional alternative references."""

    name = "exact_match"

    def verify_answer(self, task: TaskSample, candidate: str) -> VerifierOutcome:
        reference = task.reference_answer
        if task.answer_type is AnswerType.STRUCTURED:
            correct = bool(_structured_equal(reference, candidate))
            expected_normalized: Any = reference
        else:
            alternatives: Sequence[Any]
            if isinstance(reference, (list, tuple, set, frozenset)):
                alternatives = tuple(reference)
            else:
                alternatives = (reference,)
            candidate_normalized = normalize_answer(candidate)
            normalized_references = tuple(
                normalize_answer(item) for item in alternatives
            )
            correct = candidate_normalized in normalized_references
            expected_normalized = normalized_references
        return make_outcome(
            self.name,
            float(correct),
            is_correct=correct,
            details={
                "candidate": candidate,
                "candidate_normalized": normalize_answer(candidate),
                "reference_normalized": expected_normalized,
            },
        )


__all__ = ["ExactMatchVerifier", "normalize_answer", "unwrap_answer"]
