"""Unified answer verification and reward API."""

from .base import (
    AnswerOutput,
    AnswerVerifier,
    MissingReferenceError,
    VerificationError,
    extract_answer,
)
from .exact_match import ExactMatchVerifier, normalize_answer
from .llm_judge import LLMJudgeVerifier
from .multiple_choice import MultipleChoiceVerifier, resolve_choice
from .numeric import NumericVerifier, parse_number
from .router import VerifierRouter, reward_task, verify_task
from .spatial_relation import (
    SpatialRelationVerifier,
    canonical_relation,
    parse_points,
)

__all__ = [
    "AnswerOutput",
    "AnswerVerifier",
    "ExactMatchVerifier",
    "LLMJudgeVerifier",
    "MissingReferenceError",
    "MultipleChoiceVerifier",
    "NumericVerifier",
    "SpatialRelationVerifier",
    "VerificationError",
    "VerifierRouter",
    "canonical_relation",
    "extract_answer",
    "normalize_answer",
    "parse_number",
    "parse_points",
    "resolve_choice",
    "reward_task",
    "verify_task",
]
