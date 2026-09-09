"""Task-level correctness and reward accuracy metrics."""

from __future__ import annotations

from dataclasses import dataclass

from spatialcraft.schemas import Trajectory


def is_correct(trajectory: Trajectory, *, reward_threshold: float = 0.5) -> bool:
    if trajectory.verifier is not None and trajectory.verifier.is_correct is not None:
        return trajectory.verifier.is_correct
    return float(trajectory.reward or 0.0) >= reward_threshold


@dataclass(frozen=True, slots=True, kw_only=True)
class AccuracyMetrics:
    total: int
    correct: int
    accuracy: float
    mean_reward: float


def accuracy(
    trajectories: tuple[Trajectory, ...], *, reward_threshold: float = 0.5
) -> AccuracyMetrics:
    total = len(trajectories)
    correct = sum(
        is_correct(item, reward_threshold=reward_threshold) for item in trajectories
    )
    rewards = [float(item.reward or 0.0) for item in trajectories]
    return AccuracyMetrics(
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        mean_reward=sum(rewards) / total if total else 0.0,
    )


__all__ = ["AccuracyMetrics", "accuracy", "is_correct"]
