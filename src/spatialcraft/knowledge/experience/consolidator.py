"""Similarity-based Experience merge proposal generation."""

from __future__ import annotations

import re

from spatialcraft.schemas import (
    ExperienceItem,
    ExperienceOperationType,
    ExperienceProvenance,
    ExperienceUpdate,
)

from .bank import ExperienceBank


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


class ExperienceConsolidator:
    def __init__(self, *, similarity_threshold: float = 0.8) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [0, 1]")
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def similarity(first: ExperienceItem, second: ExperienceItem) -> float:
        left, right = _tokens(first.prompt_text), _tokens(second.prompt_text)
        return len(left & right) / max(1, len(left | right))

    def propose(self, bank: ExperienceBank) -> tuple[ExperienceUpdate, ...]:
        active = list(bank.active())
        consumed: set[str] = set()
        updates = []
        for index, first in enumerate(active):
            if first.reference in consumed:
                continue
            group = [first]
            for second in active[index + 1 :]:
                if (
                    second.reference not in consumed
                    and self.similarity(first, second) >= self.similarity_threshold
                ):
                    group.append(second)
            if len(group) < 2:
                continue
            consumed.update(item.reference for item in group)
            proposal = ExperienceItem(
                condition=max((item.condition for item in group), key=len),
                action=max((item.action for item in group), key=len),
                summary=f"Merged {len(group)} overlapping experiences.",
                source_experience_refs=tuple(item.reference for item in group),
                provenance=ExperienceProvenance(
                    trajectory_ids=tuple(
                        dict.fromkeys(
                            trajectory
                            for item in group
                            for trajectory in item.provenance.trajectory_ids
                        )
                    ),
                    task_ids=tuple(
                        dict.fromkeys(
                            task for item in group for task in item.provenance.task_ids
                        )
                    ),
                    datasets=tuple(
                        dict.fromkeys(
                            dataset
                            for item in group
                            for dataset in item.provenance.datasets
                        )
                    ),
                ),
            )
            updates.append(
                ExperienceUpdate(
                    operation=ExperienceOperationType.MERGE,
                    rationale=proposal.summary or "Merged similar experiences.",
                    target_experience_refs=tuple(item.reference for item in group),
                    proposed_experience=proposal,
                    source_trajectory_ids=proposal.provenance.trajectory_ids,
                )
            )
        return tuple(updates)


__all__ = ["ExperienceConsolidator"]
