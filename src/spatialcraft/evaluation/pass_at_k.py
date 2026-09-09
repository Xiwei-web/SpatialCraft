"""Unbiased pass@k estimates grouped by task."""

from __future__ import annotations

from collections import defaultdict
from math import comb

from spatialcraft.schemas import Trajectory

from .accuracy import is_correct


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n < 0 or c < 0 or c > n or k < 1:
        raise ValueError("require n >= c >= 0 and k >= 1")
    if n == 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k(trajectories: tuple[Trajectory, ...], k: int) -> float:
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[trajectory.task.task_id].append(trajectory)
    if not grouped:
        return 0.0
    estimates = [
        estimate_pass_at_k(len(rows), sum(is_correct(item) for item in rows), k)
        for rows in grouped.values()
    ]
    return sum(estimates) / len(estimates)


__all__ = ["estimate_pass_at_k", "pass_at_k"]
