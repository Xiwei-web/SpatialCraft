"""Deterministic task batching with optional isolated worker concurrency."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from spatialcraft.schemas import TaskSample, Trajectory

from .runner import RolloutRunner

RunnerFactory = Callable[[], RolloutRunner]


class BatchScheduler:
    def __init__(self, runner_factory: RunnerFactory, *, max_workers: int = 1) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.runner_factory = runner_factory
        self.max_workers = max_workers

    def _run_one(
        self, item: tuple[int, TaskSample], num_rollouts: int, seed: int
    ) -> tuple[int, tuple[Trajectory, ...]]:
        index, task = item
        result = self.runner_factory().run(
            task,
            num_rollouts=num_rollouts,
            seed=seed + index * num_rollouts,
        )
        return index, result

    def run(
        self,
        tasks: Sequence[TaskSample],
        *,
        num_rollouts: int = 1,
        seed: int = 0,
    ) -> tuple[tuple[Trajectory, ...], ...]:
        indexed = tuple(enumerate(tasks))
        if self.max_workers == 1:
            return tuple(self._run_one(item, num_rollouts, seed)[1] for item in indexed)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            outputs = tuple(
                pool.map(lambda item: self._run_one(item, num_rollouts, seed), indexed)
            )
        return tuple(result for _, result in sorted(outputs))


__all__ = ["BatchScheduler", "RunnerFactory"]
