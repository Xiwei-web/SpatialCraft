"""Answer-type dispatch and the canonical reward API."""

from __future__ import annotations

from collections.abc import Mapping

from spatialcraft.schemas import AnswerType, TaskSample, VerifierOutcome

from .base import AnswerOutput, AnswerVerifier, VerificationError
from .exact_match import ExactMatchVerifier
from .llm_judge import LLMJudgeVerifier
from .multiple_choice import MultipleChoiceVerifier
from .numeric import NumericVerifier
from .spatial_relation import SpatialRelationVerifier


class VerifierRouter:
    """Dispatch tasks to deterministic verifiers or an optional LLM judge."""

    def __init__(
        self,
        *,
        llm_judge: LLMJudgeVerifier | None = None,
        overrides: Mapping[AnswerType | str, AnswerVerifier] | None = None,
    ) -> None:
        exact = ExactMatchVerifier()
        spatial = SpatialRelationVerifier()
        self.llm_judge = llm_judge
        self._by_type: dict[AnswerType, AnswerVerifier] = {
            AnswerType.FREE_FORM: exact,
            AnswerType.MULTIPLE_CHOICE: MultipleChoiceVerifier(),
            AnswerType.SPATIAL_RELATION: spatial,
            AnswerType.NUMERIC: NumericVerifier(),
            AnswerType.BOOLEAN: exact,
            AnswerType.POINTING: spatial,
            AnswerType.STRUCTURED: exact,
        }
        for key, verifier in (overrides or {}).items():
            self._by_type[AnswerType(key)] = verifier

    def select(self, task: TaskSample) -> AnswerVerifier:
        requested = (
            str(
                task.metadata.get("verification_method")
                or task.metadata.get("verifier")
                or ""
            )
            .strip()
            .lower()
        )
        if requested in {"llm", "llm_judge"}:
            if self.llm_judge is None:
                raise VerificationError(
                    f"task {task.task_id} requests an LLM judge, but none is configured"
                )
            return self.llm_judge
        try:
            return self._by_type[task.answer_type]
        except KeyError as exc:
            raise VerificationError(
                f"no verifier registered for answer type {task.answer_type.value}"
            ) from exc

    def verify(self, task: TaskSample, output: AnswerOutput) -> VerifierOutcome:
        return self.select(task).verify(task, output)

    def reward(self, task: TaskSample, output: AnswerOutput) -> float:
        return self.verify(task, output).score


def verify_task(
    task: TaskSample,
    output: AnswerOutput,
    *,
    router: VerifierRouter | None = None,
) -> VerifierOutcome:
    return (router or VerifierRouter()).verify(task, output)


def reward_task(
    task: TaskSample,
    output: AnswerOutput,
    *,
    router: VerifierRouter | None = None,
) -> float:
    return verify_task(task, output, router=router).score


__all__ = ["VerifierRouter", "reward_task", "verify_task"]
