"""High-level façade for one configured SpatialCraft execution policy."""

from __future__ import annotations

from spatialcraft.schemas import SpatialState, TaskSample, Trajectory

from .execution_loop import ExecutionLoop


class SpatialAgent:
    def __init__(self, execution_loop: ExecutionLoop) -> None:
        self.execution_loop = execution_loop

    def solve(
        self,
        task: TaskSample,
        *,
        rollout_index: int = 0,
        random_seed: int | None = None,
        initial_state: SpatialState | None = None,
    ) -> Trajectory:
        return self.execution_loop.run(
            task,
            rollout_index=rollout_index,
            random_seed=random_seed,
            initial_state=initial_state,
        )


__all__ = ["SpatialAgent"]
