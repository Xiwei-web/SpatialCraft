"""Mutation-proof knowledge deployment with writable output sinks."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from hashlib import sha256

from spatialcraft.knowledge.coordinator import FrozenKnowledge
from spatialcraft.schemas import TaskSample, Trajectory

DeploymentSolver = Callable[[TaskSample, FrozenKnowledge], Trajectory]
TrajectorySink = Callable[[Trajectory], None]
MetricsSink = Callable[[dict], None]


def _fingerprint(bundle: FrozenKnowledge) -> str:
    payload = {
        "snapshot": bundle.snapshot.to_dict(),
        "experience_bank": bundle.experience_bank.to_dict(),
        "experience_index": bundle.experience_index.to_dict(),
        "skill_pool": bundle.skill_pool.to_dict(),
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ReadOnlyDeploymentPipeline:
    def __init__(self, knowledge: FrozenKnowledge, solver: DeploymentSolver) -> None:
        self.knowledge = knowledge
        self.solver = solver

    def run(
        self,
        tasks: Sequence[TaskSample],
        *,
        trajectory_sink: TrajectorySink | None = None,
        metrics_sink: MetricsSink | None = None,
    ) -> tuple[Trajectory, ...]:
        before = _fingerprint(self.knowledge)
        trajectories = tuple(self.solver(task, self.knowledge) for task in tasks)
        if _fingerprint(self.knowledge) != before:
            raise RuntimeError("read-only deployment mutated frozen knowledge")
        if trajectory_sink is not None:
            for trajectory in trajectories:
                trajectory_sink(trajectory)
        if metrics_sink is not None:
            scores = [float(item.reward or 0.0) for item in trajectories]
            metrics_sink(
                {
                    "snapshot_id": self.knowledge.snapshot.snapshot_id,
                    "trajectory_count": len(trajectories),
                    "mean_reward": sum(scores) / len(scores) if scores else 0.0,
                }
            )
        return trajectories


__all__ = ["ReadOnlyDeploymentPipeline"]
