"""Conflict-checked atomic coordination of Experience and Skill updates."""

from __future__ import annotations

import json
from dataclasses import dataclass

from spatialcraft.schemas import (
    ExperienceOperationType,
    ExperienceUpdate,
    KnowledgeSnapshot,
    SkillCandidate,
    SkillEvolutionType,
    SnapshotComponentKind,
    SnapshotStage,
)
from spatialcraft.storage import ManifestConflictError, SnapshotManifestStore

from .experience import ExperienceBank, ExperienceIndex
from .experience.index import Embedder
from .skill import SkillPool


class KnowledgeConflictError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenKnowledge:
    snapshot: KnowledgeSnapshot
    experience_bank: ExperienceBank
    experience_index: ExperienceIndex
    skill_pool: SkillPool

    def __post_init__(self) -> None:
        if not self.experience_bank.frozen or not self.skill_pool.frozen:
            raise ValueError("knowledge banks must be frozen at a batch boundary")
        if not self.experience_index.frozen:
            raise ValueError("embedding index must be frozen at a batch boundary")


class KnowledgeCoordinator:
    def __init__(self, manifests: SnapshotManifestStore, embedder: Embedder) -> None:
        self.manifests = manifests
        self.embedder = embedder

    def _components(
        self,
        bank: ExperienceBank,
        index: ExperienceIndex,
        skills: SkillPool,
    ):
        return (
            self.manifests.store_component_json(
                kind=SnapshotComponentKind.EXPERIENCE_BANK,
                value=bank.freeze().to_dict(),
                item_count=len(bank.active()),
            ),
            self.manifests.store_component_json(
                kind=SnapshotComponentKind.EXPERIENCE_INDEX,
                value=index.to_dict(),
                item_count=len(index.references),
            ),
            self.manifests.store_component_json(
                kind=SnapshotComponentKind.SKILL_POOL,
                value=skills.freeze().to_dict(),
                item_count=len(skills.active()),
            ),
        )

    def initialize(
        self,
        *,
        run_id: str,
        experience_bank: ExperienceBank | None = None,
        skill_pool: SkillPool | None = None,
        publish_as: str = "latest",
    ) -> FrozenKnowledge:
        bank = experience_bank or ExperienceBank()
        skills = skill_pool or SkillPool()
        index = ExperienceIndex.build(bank, self.embedder)
        snapshot = KnowledgeSnapshot(
            run_id=run_id,
            iteration=0,
            stage=SnapshotStage.INITIAL,
            components=self._components(bank, index, skills),
            metadata={"batch_barrier": True},
        )
        self.manifests.publish(snapshot, name=publish_as)
        return self.load(snapshot)

    def load(self, snapshot: KnowledgeSnapshot) -> FrozenKnowledge:
        paths = self.manifests.verify(snapshot)
        bank = ExperienceBank.from_dict(
            json.loads(paths[SnapshotComponentKind.EXPERIENCE_BANK].read_text()),
            frozen=True,
        )
        index = ExperienceIndex.from_dict(
            json.loads(paths[SnapshotComponentKind.EXPERIENCE_INDEX].read_text())
        ).freeze()
        skills = SkillPool.from_dict(
            json.loads(paths[SnapshotComponentKind.SKILL_POOL].read_text()), frozen=True
        )
        return FrozenKnowledge(
            snapshot=snapshot,
            experience_bank=bank,
            experience_index=index,
            skill_pool=skills,
        )

    def load_current(self, run_id: str, *, name: str = "latest") -> FrozenKnowledge:
        return self.load(self.manifests.resolve_pointer(run_id, name))

    @staticmethod
    def _check_experience_conflicts(updates: tuple[ExperienceUpdate, ...]) -> None:
        claimed: set[str] = set()
        proposed_ids: set[str] = set()
        for update in updates:
            overlap = claimed & set(update.target_experience_refs)
            if overlap:
                raise KnowledgeConflictError(
                    f"multiple Experience updates target {sorted(overlap)}"
                )
            claimed.update(update.target_experience_refs)
            if update.proposed_experience is not None:
                identifier = update.proposed_experience.experience_id
                if (
                    update.operation
                    in {ExperienceOperationType.ADD, ExperienceOperationType.MERGE}
                    and identifier in proposed_ids
                ):
                    raise KnowledgeConflictError(
                        f"multiple Experience proposals create id {identifier}"
                    )
                proposed_ids.add(identifier)

    @staticmethod
    def _check_skill_conflicts(candidates: tuple[SkillCandidate, ...]) -> None:
        parents: set[str] = set()
        new_ids: set[str] = set()
        for candidate in candidates:
            if candidate.evolution_type is SkillEvolutionType.REFINE:
                parent = candidate.skill.parent_skill_ref
                assert parent is not None
                if parent in parents:
                    raise KnowledgeConflictError(
                        f"multiple accepted candidates refine {parent}"
                    )
                parents.add(parent)
            else:
                if candidate.skill.skill_id in new_ids:
                    raise KnowledgeConflictError(
                        f"multiple accepted candidates create {candidate.skill.skill_id}"
                    )
                new_ids.add(candidate.skill.skill_id)

    def commit_batch(
        self,
        base: FrozenKnowledge,
        *,
        experience_updates: tuple[ExperienceUpdate, ...] = (),
        accepted_skill_candidates: tuple[SkillCandidate, ...] = (),
        pruned_skill_references: tuple[str, ...] = (),
        source_trajectory_ids: tuple[str, ...],
        publish_as: str = "latest",
    ) -> FrozenKnowledge:
        current = self.manifests.resolve_pointer(base.snapshot.run_id, publish_as)
        if current.snapshot_id != base.snapshot.snapshot_id:
            raise KnowledgeConflictError(
                "batch base snapshot is stale; refusing a non-reproducible commit"
            )
        self._check_experience_conflicts(experience_updates)
        self._check_skill_conflicts(accepted_skill_candidates)
        bank = ExperienceBank(base.experience_bank.all()).apply_many(experience_updates)
        skills = SkillPool(base.skill_pool.all())
        for candidate in accepted_skill_candidates:
            skills = (
                skills.refine(candidate.skill)
                if candidate.evolution_type is SkillEvolutionType.REFINE
                else skills.add(candidate.skill)
            )
        for reference in pruned_skill_references:
            skills = skills.archive(reference)
        index = ExperienceIndex.build(bank, self.embedder)
        snapshot = KnowledgeSnapshot(
            run_id=base.snapshot.run_id,
            iteration=base.snapshot.iteration + 1,
            stage=SnapshotStage.ACCUMULATION,
            components=self._components(bank, index, skills),
            parent_snapshot_id=base.snapshot.snapshot_id,
            source_trajectory_ids=tuple(dict.fromkeys(source_trajectory_ids)),
            metadata={
                "batch_barrier": True,
                "experience_update_ids": [
                    item.update_id for item in experience_updates
                ],
                "accepted_skill_candidate_ids": [
                    item.candidate_id for item in accepted_skill_candidates
                ],
                "pruned_skill_references": list(pruned_skill_references),
            },
        )
        try:
            self.manifests.publish(snapshot, name=publish_as)
        except ManifestConflictError as exc:
            raise KnowledgeConflictError(
                "concurrent snapshot publication conflict"
            ) from exc
        return self.load(snapshot)


__all__ = ["FrozenKnowledge", "KnowledgeConflictError", "KnowledgeCoordinator"]
