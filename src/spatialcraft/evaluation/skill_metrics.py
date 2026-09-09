"""Skill activation, persistence, gain, and gate metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from spatialcraft.schemas import PPOGateRecord, Trajectory


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillMetrics:
    trajectory_count: int
    activation_rate: float
    active_transition_rate: float
    mean_active_lifetime: float
    unique_skill_versions: int
    activation_frequency: dict[str, int]
    ppo_candidate_count: int
    ppo_acceptance_rate: float


def skill_metrics(
    trajectories: tuple[Trajectory, ...],
    *,
    gate_records: tuple[PPOGateRecord, ...] = (),
) -> SkillMetrics:
    frequencies: Counter[str] = Counter()
    lifetimes = []
    active_transitions = 0
    total_transitions = sum(len(item.transitions) for item in trajectories)
    with_skill = 0
    for trajectory in trajectories:
        runs: list[tuple[str, int]] = []
        current = None
        length = 0
        for transition in trajectory.transitions:
            ref = (
                f"{transition.active_skill.skill_id}@{transition.active_skill.version}"
                if transition.active_skill
                else None
            )
            if ref is not None:
                active_transitions += 1
                frequencies[ref] += 1
            if ref == current:
                length += int(ref is not None)
            else:
                if current is not None:
                    runs.append((current, length))
                current, length = ref, int(ref is not None)
        if current is not None:
            runs.append((current, length))
        if runs:
            with_skill += 1
            lifetimes.extend(value for _, value in runs)
    return SkillMetrics(
        trajectory_count=len(trajectories),
        activation_rate=with_skill / len(trajectories) if trajectories else 0.0,
        active_transition_rate=(
            active_transitions / total_transitions if total_transitions else 0.0
        ),
        mean_active_lifetime=sum(lifetimes) / len(lifetimes) if lifetimes else 0.0,
        unique_skill_versions=len(frequencies),
        activation_frequency=dict(sorted(frequencies.items())),
        ppo_candidate_count=len(gate_records),
        ppo_acceptance_rate=(
            sum(item.accepted for item in gate_records) / len(gate_records)
            if gate_records
            else 0.0
        ),
    )


__all__ = ["SkillMetrics", "skill_metrics"]
