"""Atomic Experience Bank/Index publication as a KnowledgeSnapshot."""

from __future__ import annotations

from spatialcraft.schemas import (
    KnowledgeSnapshot,
    SnapshotComponentKind,
    SnapshotStage,
)
from spatialcraft.storage import SnapshotManifestStore

from .bank import ExperienceBank
from .index import ExperienceIndex


class ExperienceSnapshotStore:
    def __init__(self, manifests: SnapshotManifestStore) -> None:
        self.manifests = manifests

    def commit(
        self,
        *,
        run_id: str,
        iteration: int,
        bank: ExperienceBank,
        index: ExperienceIndex,
        parent_snapshot_id: str | None = None,
        skill_pool_value: dict | list | None = None,
        source_trajectory_ids: tuple[str, ...] = (),
        publish_as: str = "latest",
    ) -> KnowledgeSnapshot:
        if (iteration == 0) != (parent_snapshot_id is None):
            raise ValueError(
                "iteration zero requires no parent; later iterations require a parent"
            )
        bank_ref = self.manifests.store_component_json(
            kind=SnapshotComponentKind.EXPERIENCE_BANK,
            value=bank.freeze().to_dict(),
            item_count=len(bank.active()),
        )
        index_ref = self.manifests.store_component_json(
            kind=SnapshotComponentKind.EXPERIENCE_INDEX,
            value=index.to_dict(),
            item_count=len(index.references),
        )
        skill_value = (
            skill_pool_value
            if skill_pool_value is not None
            else {
                "schema_version": "1.0",
                "items": [],
            }
        )
        skill_ref = self.manifests.store_component_json(
            kind=SnapshotComponentKind.SKILL_POOL,
            value=skill_value,
            item_count=len(skill_value.get("items", ()))
            if isinstance(skill_value, dict)
            else len(skill_value),
        )
        snapshot = KnowledgeSnapshot(
            run_id=run_id,
            iteration=iteration,
            stage=(
                SnapshotStage.INITIAL if iteration == 0 else SnapshotStage.ACCUMULATION
            ),
            components=(bank_ref, index_ref, skill_ref),
            parent_snapshot_id=parent_snapshot_id,
            source_trajectory_ids=tuple(dict.fromkeys(source_trajectory_ids)),
            metadata={"experience_count": len(bank.active())},
        )
        self.manifests.publish(snapshot, name=publish_as)
        return snapshot

    def load(
        self, snapshot: KnowledgeSnapshot
    ) -> tuple[ExperienceBank, ExperienceIndex]:
        verified = self.manifests.verify(snapshot)
        import json

        bank_value = json.loads(
            verified[SnapshotComponentKind.EXPERIENCE_BANK].read_text(encoding="utf-8")
        )
        index_value = json.loads(
            verified[SnapshotComponentKind.EXPERIENCE_INDEX].read_text(encoding="utf-8")
        )
        return (
            ExperienceBank.from_dict(bank_value, frozen=True),
            ExperienceIndex.from_dict(index_value),
        )


__all__ = ["ExperienceSnapshotStore"]
