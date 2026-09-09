"""Common verifier contracts and answer extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, TypeAlias

from spatialcraft.models import ModelResponse
from spatialcraft.schemas import ActionType, AgentAction, TaskSample, VerifierOutcome

AnswerOutput: TypeAlias = str | AgentAction | ModelResponse | Mapping[str, Any] | None


class VerificationError(RuntimeError):
    """Base error raised by the verification layer."""


class MissingReferenceError(VerificationError):
    """Raised when evaluation is attempted on a task without a reference."""


def extract_answer(output: AnswerOutput) -> str | None:
    """Extract a candidate answer from supported executor output types."""

    value: Any
    if output is None:
        return None
    if isinstance(output, str):
        value = output
    elif isinstance(output, AgentAction):
        if output.action_type is not ActionType.FINAL:
            return None
        value = output.final_answer
    elif isinstance(output, ModelResponse):
        value = output.text
    elif isinstance(output, Mapping):
        value = next(
            (
                output[key]
                for key in ("final_answer", "answer", "text", "output")
                if key in output and output[key] is not None
            ),
            None,
        )
    else:
        raise TypeError(f"unsupported answer output type: {type(output).__name__}")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def make_outcome(
    verifier_name: str,
    score: float,
    *,
    is_correct: bool | None = None,
    analysis: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> VerifierOutcome:
    """Build an outcome while enforcing the project-wide reward range."""

    value = float(score)
    if not 0.0 <= value <= 1.0:
        raise ValueError("verifier rewards must be in [0, 1]")
    return VerifierOutcome(
        verifier_name=verifier_name,
        score=value,
        is_correct=is_correct,
        analysis=analysis,
        details=dict(details or {}),
    )


class AnswerVerifier(ABC):
    """Base class for deterministic and model-based answer verifiers."""

    name: str

    def verify(self, task: TaskSample, output: AnswerOutput) -> VerifierOutcome:
        if task.reference_answer is None:
            raise MissingReferenceError(
                f"task {task.task_id} does not include a reference answer"
            )
        candidate = extract_answer(output)
        if candidate is None:
            return make_outcome(
                self.name,
                0.0,
                is_correct=False,
                analysis="No final answer was produced.",
                details={"reason": "missing_candidate"},
            )
        return self.verify_answer(task, candidate)

    @abstractmethod
    def verify_answer(self, task: TaskSample, candidate: str) -> VerifierOutcome:
        """Compare one non-empty candidate against the task reference."""


__all__ = [
    "AnswerOutput",
    "AnswerVerifier",
    "MissingReferenceError",
    "VerificationError",
    "extract_answer",
    "make_outcome",
]
