"""State-conditioned deterministic skill-or-NONE selection."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from spatialcraft.knowledge.experience.index import Embedder, _normalize
from spatialcraft.schemas import SkillItem, SpatialState, TaskSample

from .pool import SkillPool

SelectionScorer = Callable[[SkillItem, TaskSample, SpatialState], float]


class SkillSelector:
    def __init__(
        self,
        *,
        enabled: bool = True,
        minimum_score: float | None = None,
        scorer: SelectionScorer | None = None,
        embedder: Embedder | None = None,
        applicability_judge: Callable | None = None,
    ) -> None:
        self.enabled = enabled
        self.minimum_score = minimum_score
        self.scorer = scorer
        self.embedder = embedder
        self.applicability_judge = applicability_judge

    @staticmethod
    def _heuristic(skill: SkillItem, task: TaskSample, state: SpatialState) -> float:
        text = " ".join(
            (
                task.question,
                " ".join(task.choices),
                " ".join(item.summary for item in state.evidence),
            )
        ).lower()
        keywords = tuple(
            str(value).lower() for value in skill.metadata.get("keywords", ())
        )
        score = float(skill.metadata.get("priority", 0.0))
        score += sum(1.0 for keyword in keywords if keyword and keyword in text)
        if skill.name == "StrategicPlanning" and (
            len(task.images) > 1 or state.step_index == 0
        ):
            score += 0.5
        if skill.name == "ReActDecision" and state.step_index > 0:
            score += 0.25
        return score

    def select(
        self,
        pool: SkillPool,
        task: TaskSample,
        state: SpatialState,
        *,
        excluded_references: frozenset[str] = frozenset(),
    ) -> SkillItem | None:
        if not self.enabled:
            return None
        candidates = [
            item for item in pool.active() if item.reference not in excluded_references
        ]
        if not candidates:
            return None
        if self.applicability_judge is not None:
            references = tuple(self.applicability_judge(tuple(candidates), task, state))
            available = {item.reference for item in candidates}
            if (
                len(references) != len(set(references))
                or not set(references) <= available
            ):
                raise ValueError(
                    "Applicability judge returned duplicate or unknown Skill references"
                )
            candidates = [item for item in candidates if item.reference in references]
            if not candidates:
                return None
        if self.embedder is not None:
            query = "\n".join(
                (
                    task.question,
                    *(e.summary for e in state.evidence),
                )
            )
            vectors = _normalize(
                self.embedder.embed([query, *(s.initiation for s in candidates)])
            )
            scores = vectors[1:] @ vectors[0]
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite Skill similarity")
            # Exactly one argmax; no top-3 activation and no inherited SMA threshold.
            score, selected = max(
                zip(scores, candidates, strict=True),
                key=lambda pair: (float(pair[0]), pair[1].reference),
            )
            return (
                selected
                if self.minimum_score is None or float(score) >= self.minimum_score
                else None
            )
        scorer = self.scorer or self._heuristic
        ranked = sorted(
            ((float(scorer(item, task, state)), item) for item in candidates),
            key=lambda pair: (-pair[0], pair[1].reference),
        )
        return (
            ranked[0][1]
            if ranked[0][0]
            >= (0.0 if self.minimum_score is None else self.minimum_score)
            else None
        )


__all__ = ["SelectionScorer", "SkillSelector"]
