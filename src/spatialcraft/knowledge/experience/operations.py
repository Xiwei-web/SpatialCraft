"""Generate and apply auditable Experience Add/Modify operations."""

from __future__ import annotations

from dataclasses import replace

from spatialcraft.schemas import (
    ExperienceItem,
    ExperienceOperationType,
    ExperienceProvenance,
    ExperienceUpdate,
)

from .bank import ExperienceBank
from .cross_rollout_critic import RolloutCritique


def update_from_critique(
    critique: RolloutCritique,
    *,
    existing: ExperienceItem | None = None,
    dataset: str | None = None,
) -> ExperienceUpdate:
    trajectories = tuple(
        dict.fromkeys((*critique.best_trajectory_ids, *critique.failed_trajectory_ids))
    )
    provenance = ExperienceProvenance(
        trajectory_ids=trajectories,
        task_ids=(critique.task_id,),
        datasets=(dataset,) if dataset else (),
    )
    if existing is None:
        proposal = ExperienceItem(
            condition=critique.reusable_condition,
            action=critique.reusable_action,
            summary=critique.rationale,
            provenance=provenance,
            metadata={"post_rollout_only": True},
        )
        return ExperienceUpdate(
            operation=ExperienceOperationType.ADD,
            rationale=critique.rationale,
            proposed_experience=proposal,
            source_trajectory_ids=trajectories,
        )
    proposal = replace(
        existing,
        version=existing.version + 1,
        condition=critique.reusable_condition,
        action=critique.reusable_action,
        summary=critique.rationale,
        provenance=provenance,
        source_experience_refs=(existing.reference,),
    )
    return ExperienceUpdate(
        operation=ExperienceOperationType.MODIFY,
        rationale=critique.rationale,
        target_experience_refs=(existing.reference,),
        proposed_experience=proposal,
        source_trajectory_ids=trajectories,
    )


def apply_updates(
    bank: ExperienceBank, updates: tuple[ExperienceUpdate, ...]
) -> ExperienceBank:
    return bank.apply_many(updates)


__all__ = ["apply_updates", "update_from_critique"]
