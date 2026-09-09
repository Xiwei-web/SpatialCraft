"""Experience utility accounting and low-value archival proposals."""

from __future__ import annotations

from dataclasses import replace

from spatialcraft.schemas import (
    ExperienceOperationType,
    ExperienceUpdate,
    Trajectory,
)

from .bank import ExperienceBank


class ExperienceMaintenance:
    def enforce_capacity(self, bank: ExperienceBank, *, capacity: int = 100):
        """Archive redundant/low-utility items; preserve all historical records."""
        from .consolidator import ExperienceConsolidator

        if bank.frozen:
            raise ValueError("Capacity maintenance cannot modify deployment knowledge")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        working, updates = bank, []
        if len(working.active()) <= capacity:
            return working, ()
        # Only exactly matching conditional lessons are safe to combine without
        # a semantic merge judge. Nonidentical items are pruned, not spliced.
        for update in ExperienceConsolidator(similarity_threshold=1.0).propose(working):
            targets = [working.get(ref) for ref in update.target_experience_refs]
            if len({(item.condition, item.action) for item in targets}) != 1:
                continue
            working = working.apply(update)
            updates.append(update)
            if len(working.active()) <= capacity:
                break
        excess = len(working.active()) - capacity
        if excess > 0:
            ranked = sorted(
                working.active(),
                key=lambda item: (
                    item.stats.average_reward_gain,
                    item.stats.use_count,
                    item.updated_at,
                    item.reference,
                ),
            )
            update = ExperienceUpdate(
                operation=ExperienceOperationType.ARCHIVE,
                rationale=f"Capacity={capacity}: archive lowest utility/least-used items.",
                target_experience_refs=tuple(
                    item.reference for item in ranked[:excess]
                ),
            )
            working = working.apply(update)
            updates.append(update)
        return working, tuple(updates)

    def record_trajectory_usage(
        self, bank: ExperienceBank, trajectory: Trajectory, *, baseline: float = 0.0
    ) -> ExperienceBank:
        if bank.frozen:
            raise ValueError("usage statistics require an unfrozen Experience Bank")
        used = set(trajectory.used_experience_ids)
        values = []
        gain = float(trajectory.reward or 0.0) - baseline
        for item in bank.all():
            if item.experience_id in used and item.status.value == "active":
                stats = item.stats.record_retrieval().record_use(
                    reward_gain=gain, successful=gain > 0
                )
                values.append(replace(item, stats=stats))
            else:
                values.append(item)
        return ExperienceBank(tuple(values))

    def propose_archives(
        self,
        bank: ExperienceBank,
        *,
        minimum_uses: int = 3,
        maximum_average_gain: float = 0.0,
    ) -> tuple[ExperienceUpdate, ...]:
        targets = tuple(
            item.reference
            for item in bank.active()
            if item.stats.use_count >= minimum_uses
            and item.stats.average_reward_gain <= maximum_average_gain
        )
        if not targets:
            return ()
        return (
            ExperienceUpdate(
                operation=ExperienceOperationType.ARCHIVE,
                rationale=(
                    f"Archive experiences with at least {minimum_uses} uses and "
                    f"average gain <= {maximum_average_gain}."
                ),
                target_experience_refs=targets,
            ),
        )


__all__ = ["ExperienceMaintenance"]
