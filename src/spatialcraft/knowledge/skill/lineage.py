"""Auditable parent/version lineage for evolved procedural skills."""

from __future__ import annotations

from dataclasses import dataclass

from spatialcraft.schemas import SkillCandidate


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillLineageRecord:
    child_reference: str
    parent_reference: str | None
    candidate_id: str
    source_gradient_ids: tuple[str, ...]
    source_trajectory_ids: tuple[str, ...]


class SkillLineage:
    def __init__(self, records: tuple[SkillLineageRecord, ...] = ()) -> None:
        if len({item.child_reference for item in records}) != len(records):
            raise ValueError(
                "each child skill version can have only one lineage record"
            )
        self._records = {item.child_reference: item for item in records}
        for reference in self._records:
            self.ancestors(reference)

    def record(self, candidate: SkillCandidate) -> SkillLineage:
        child = candidate.skill.reference
        if child in self._records:
            raise ValueError(f"lineage already records {child}")
        item = SkillLineageRecord(
            child_reference=child,
            parent_reference=candidate.skill.parent_skill_ref,
            candidate_id=candidate.candidate_id,
            source_gradient_ids=candidate.source_gradient_ids,
            source_trajectory_ids=candidate.source_trajectory_ids,
        )
        return SkillLineage((*self.records(), item))

    def records(self) -> tuple[SkillLineageRecord, ...]:
        return tuple(
            sorted(self._records.values(), key=lambda item: item.child_reference)
        )

    def ancestors(self, reference: str) -> tuple[str, ...]:
        seen = {reference}
        output = []
        current = self._records.get(reference)
        while current is not None and current.parent_reference is not None:
            parent = current.parent_reference
            if parent in seen:
                raise ValueError("skill lineage contains a cycle")
            seen.add(parent)
            output.append(parent)
            current = self._records.get(parent)
        return tuple(output)


__all__ = ["SkillLineage", "SkillLineageRecord"]
