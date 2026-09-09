"""Single-task multi-rollout orchestration."""

from __future__ import annotations

from spatialcraft.agent import SpatialAgent
from spatialcraft.schemas import TaskSample, Trajectory

from .recorder import TrajectoryRecorder


class RolloutRunner:
    def __init__(
        self, agent: SpatialAgent, *, recorder: TrajectoryRecorder | None = None
    ) -> None:
        self.agent = agent
        self.recorder = recorder

    def run(
        self,
        task: TaskSample,
        *,
        num_rollouts: int = 1,
        seed: int = 0,
        start_index: int = 0,
    ) -> tuple[Trajectory, ...]:
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be positive")
        trajectories = tuple(
            self.agent.solve(
                task,
                rollout_index=start_index + index,
                random_seed=seed + index,
            )
            for index in range(num_rollouts)
        )
        if self.recorder is not None:
            for trajectory in trajectories:
                self.recorder.append(trajectory)
        return trajectories


__all__ = ["RolloutRunner"]
