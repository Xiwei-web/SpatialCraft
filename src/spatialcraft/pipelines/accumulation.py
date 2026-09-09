"""Batch-barrier accumulation pipeline for reproducible knowledge evolution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from spatialcraft.knowledge.coordinator import FrozenKnowledge, KnowledgeCoordinator
from spatialcraft.schemas import (
    ExperienceUpdate,
    SkillCandidate,
    TaskSample,
    Trajectory,
)

RolloutBatch = Callable[
    [tuple[TaskSample, ...], FrozenKnowledge, int, int], tuple[Trajectory, ...]
]
RewardFunction = Callable[[Trajectory], float]
ExperienceBuilder = Callable[
    [FrozenKnowledge, tuple[Trajectory, ...]], tuple[ExperienceUpdate, ...]
]
SkillBuilder = Callable[
    [FrozenKnowledge, tuple[Trajectory, ...]], tuple[SkillCandidate, ...]
]
SkillPruner = Callable[[FrozenKnowledge, tuple[Trajectory, ...]], tuple[str, ...]]


@dataclass(frozen=True, slots=True, kw_only=True)
class AccumulationResult:
    base_snapshot_id: str
    committed_snapshot_id: str
    trajectories: tuple[Trajectory, ...]
    experience_updates: tuple[ExperienceUpdate, ...]
    accepted_skill_candidates: tuple[SkillCandidate, ...]
    phases: tuple[str, ...]


class AccumulationPipeline:
    def __init__(
        self,
        coordinator: KnowledgeCoordinator,
        rollout_batch: RolloutBatch,
        *,
        reward_function: RewardFunction | None = None,
        experience_builder: ExperienceBuilder | None = None,
        skill_builder: SkillBuilder | None = None,
        skill_pruner: SkillPruner | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.rollout_batch = rollout_batch
        self.reward_function = reward_function
        self.experience_builder = experience_builder or (lambda bundle, rows: ())
        self.skill_builder = skill_builder or (lambda bundle, rows: ())
        self.skill_pruner = skill_pruner or (lambda bundle, rows: ())

    @staticmethod
    def _set_reward(trajectory: Trajectory, reward: float) -> Trajectory:
        transitions = trajectory.transitions
        if transitions:
            transitions = (*transitions[:-1], replace(transitions[-1], reward=reward))
        return replace(trajectory, reward=float(reward), transitions=transitions)

    def run(
        self,
        *,
        run_id: str,
        tasks: Sequence[TaskSample],
        num_rollouts: int = 1,
        seed: int = 0,
        pointer_name: str = "latest",
    ) -> AccumulationResult:
        if not tasks or num_rollouts < 1:
            raise ValueError("accumulation requires tasks and positive num_rollouts")
        phases = ["freeze_snapshot"]
        base = self.coordinator.load_current(run_id, name=pointer_name)
        trajectories = tuple(self.rollout_batch(tuple(tasks), base, num_rollouts, seed))
        phases.append("rollouts_complete")
        if any(
            item.knowledge_snapshot_id != base.snapshot.snapshot_id
            for item in trajectories
        ):
            raise ValueError("all batch rollouts must use the frozen base snapshot")
        if self.reward_function is not None:
            trajectories = tuple(
                self._set_reward(item, self.reward_function(item))
                for item in trajectories
            )
        elif any(item.reward is None for item in trajectories):
            raise ValueError("all trajectories require reward after the batch barrier")
        phases.append("rewards_complete")
        experience_updates = self.experience_builder(base, trajectories)
        phases.append("experience_updates_built")
        skill_candidates = self.skill_builder(base, trajectories)
        pruned = self.skill_pruner(base, trajectories)
        phases.append("skill_updates_built")
        committed = self.coordinator.commit_batch(
            base,
            experience_updates=experience_updates,
            accepted_skill_candidates=skill_candidates,
            pruned_skill_references=pruned,
            source_trajectory_ids=tuple(item.trajectory_id for item in trajectories),
            publish_as=pointer_name,
        )
        phases.append("snapshot_committed")
        return AccumulationResult(
            base_snapshot_id=base.snapshot.snapshot_id,
            committed_snapshot_id=committed.snapshot.snapshot_id,
            trajectories=trajectories,
            experience_updates=experience_updates,
            accepted_skill_candidates=skill_candidates,
            phases=tuple(phases),
        )


__all__ = ["AccumulationPipeline", "AccumulationResult"]
