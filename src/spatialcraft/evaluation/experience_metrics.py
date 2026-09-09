"""Experience retrieval/use coverage and reward-lift metrics."""

from __future__ import annotations

from dataclasses import dataclass

from spatialcraft.schemas import Trajectory


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperienceMetrics:
    trajectory_count: int
    trajectories_with_experience: int
    use_rate: float
    unique_experiences_used: int
    mean_experiences_per_trajectory: float
    reward_lift: float | None


def experience_metrics(
    trajectories: tuple[Trajectory, ...],
    *,
    vanilla: tuple[Trajectory, ...] = (),
) -> ExperienceMetrics:
    counts = [len(item.used_experience_ids) for item in trajectories]
    used = {
        reference
        for trajectory in trajectories
        for reference in trajectory.used_experience_ids
    }
    mean_reward = (
        sum(float(item.reward or 0.0) for item in trajectories) / len(trajectories)
        if trajectories
        else 0.0
    )
    vanilla_reward = (
        sum(float(item.reward or 0.0) for item in vanilla) / len(vanilla)
        if vanilla
        else None
    )
    return ExperienceMetrics(
        trajectory_count=len(trajectories),
        trajectories_with_experience=sum(count > 0 for count in counts),
        use_rate=sum(count > 0 for count in counts) / len(counts) if counts else 0.0,
        unique_experiences_used=len(used),
        mean_experiences_per_trajectory=sum(counts) / len(counts) if counts else 0.0,
        reward_lift=mean_reward - vanilla_reward
        if vanilla_reward is not None
        else None,
    )


__all__ = ["ExperienceMetrics", "experience_metrics"]
