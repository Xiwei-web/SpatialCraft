"""Task-baseline advantages and exact trajectory-to-skill credit assignment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

from spatialcraft.schemas import Trajectory

from .segments import skill_segments


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillCredit:
    trajectory_id: str
    skill_reference: str
    advantage: float
    activation_steps: int
    successful: bool
    activation_start_step: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryAdvantage:
    trajectory_id: str
    task_id: str
    reward: float
    baseline: float
    advantage: float


class SkillCreditAssigner:
    """Use a shared per-task baseline and attribute only actually active skills."""

    @staticmethod
    def task_baselines(trajectories: tuple[Trajectory, ...]) -> dict[str, float]:
        rewards: dict[str, list[float]] = defaultdict(list)
        for trajectory in trajectories:
            if trajectory.reward is not None:
                rewards[trajectory.task.task_id].append(float(trajectory.reward))
        return {
            task_id: sum(values) / len(values) for task_id, values in rewards.items()
        }

    def advantages(
        self,
        trajectories: tuple[Trajectory, ...],
        *,
        baselines: dict[str, float] | None = None,
    ) -> tuple[TrajectoryAdvantage, ...]:
        resolved = dict(baselines or self.task_baselines(trajectories))
        output = []
        for trajectory in trajectories:
            reward = float(trajectory.reward or 0.0)
            baseline = float(resolved.get(trajectory.task.task_id, 0.0))
            if not isfinite(baseline):
                raise ValueError("task baselines must be finite")
            output.append(
                TrajectoryAdvantage(
                    trajectory_id=trajectory.trajectory_id,
                    task_id=trajectory.task.task_id,
                    reward=reward,
                    baseline=baseline,
                    advantage=reward - baseline,
                )
            )
        return tuple(output)

    def assign(
        self,
        trajectories: tuple[Trajectory, ...],
        *,
        baselines: dict[str, float] | None = None,
    ) -> tuple[SkillCredit, ...]:
        advantage_by_id = {
            item.trajectory_id: item
            for item in self.advantages(trajectories, baselines=baselines)
        }
        credits: list[SkillCredit] = []
        for trajectory in trajectories:
            advantage = advantage_by_id[trajectory.trajectory_id].advantage
            for segment in skill_segments(trajectory):
                if segment.skill_reference is None:
                    continue
                credits.append(
                    SkillCredit(
                        trajectory_id=trajectory.trajectory_id,
                        skill_reference=segment.skill_reference,
                        # Draft Eq.9: per-step advantage normalized by trajectory
                        # length, then averaged within this activation window.
                        advantage=advantage / max(1, len(trajectory.transitions)),
                        activation_steps=len(segment.transitions),
                        successful=advantage > 0.0,
                        activation_start_step=segment.start_step,
                    )
                )
        return tuple(credits)


__all__ = ["SkillCredit", "SkillCreditAssigner", "TrajectoryAdvantage"]
