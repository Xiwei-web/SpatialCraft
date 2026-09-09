"""Top-K per-subtask Experience retrieval with deterministic deduplication."""

from __future__ import annotations

from collections import defaultdict

from spatialcraft.schemas import RetrievedExperienceRef, TaskSample

from .bank import ExperienceBank
from .index import Embedder, ExperienceIndex
from .task_decomposer import TaskDecomposer


class ExperienceRetriever:
    def __init__(
        self,
        bank: ExperienceBank,
        index: ExperienceIndex,
        embedder: Embedder,
        *,
        decomposer: TaskDecomposer | None = None,
    ) -> None:
        self.bank = bank
        self.index = index
        self.embedder = embedder
        self.decomposer = decomposer or TaskDecomposer()

    def retrieve(
        self,
        task: TaskSample,
        *,
        top_k: int | None = None,
        per_subtask: int = 3,
        minimum_score: float = -1.0,
    ) -> tuple[RetrievedExperienceRef, ...]:
        if (top_k is not None and top_k < 1) or per_subtask < 1:
            raise ValueError("retrieval limits must be positive")
        matches: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for subtask in self.decomposer.decompose(task.without_reference_answer()):
            for reference, score in self.index.search(
                subtask, self.embedder, top_k=per_subtask
            ):
                if score >= minimum_score:
                    matches[reference].append((subtask, score))
        ranked = sorted(
            matches,
            key=lambda reference: (
                -max(score for _, score in matches[reference]),
                reference,
            ),
        )[:top_k]
        return tuple(
            RetrievedExperienceRef(
                experience_id=(item := self.bank.get(reference)).experience_id,
                version=item.version,
                retrieval_score=max(
                    -1.0, min(1.0, max(score for _, score in matches[reference]))
                ),
                original_text=item.prompt_text,
                matched_subtasks=tuple(subtask for subtask, _ in matches[reference]),
            )
            for reference in ranked
        )


__all__ = ["ExperienceRetriever"]
