"""Spatial tool utilization, success, and diversity metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from spatialcraft.schemas import Trajectory


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolMetrics:
    trajectory_count: int
    total_calls: int
    successful_calls: int
    failed_calls: int
    success_rate: float
    calls_per_trajectory: float
    unique_tools: int
    call_frequency: dict[str, int]


def tool_metrics(trajectories: tuple[Trajectory, ...]) -> ToolMetrics:
    results = [
        result
        for trajectory in trajectories
        for transition in trajectory.transitions
        for result in transition.tool_results
    ]
    frequency = Counter(item.tool_name for item in results)
    successful = sum(item.succeeded for item in results)
    total = len(results)
    count = len(trajectories)
    return ToolMetrics(
        trajectory_count=count,
        total_calls=total,
        successful_calls=successful,
        failed_calls=total - successful,
        success_rate=successful / total if total else 0.0,
        calls_per_trajectory=total / count if count else 0.0,
        unique_tools=len(frequency),
        call_frequency=dict(sorted(frequency.items())),
    )


__all__ = ["ToolMetrics", "tool_metrics"]
