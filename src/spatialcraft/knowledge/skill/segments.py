"""Actual contiguous skill activation windows, never whole-trajectory attribution."""

from __future__ import annotations

from dataclasses import dataclass

from spatialcraft.schemas import TaskSample, Trajectory, Transition


@dataclass(frozen=True, slots=True)
class SkillSegment:
    trajectory_id: str
    task: TaskSample
    reward: float | None
    skill_reference: str | None
    start_step: int
    end_step: int
    transitions: tuple[Transition, ...]


def skill_segments(trajectory: Trajectory) -> tuple[SkillSegment, ...]:
    groups: list[list[Transition]] = []
    previous = object()
    for transition in trajectory.transitions:
        active = transition.active_skill
        key = (
            (active.skill_id, active.version, active.activated_at_step)
            if active
            else None
        )
        if key != previous:
            groups.append([])
        groups[-1].append(transition)
        previous = key
    return tuple(
        SkillSegment(
            trajectory.trajectory_id,
            trajectory.task,
            trajectory.reward,
            f"{rows[0].active_skill.skill_id}@{rows[0].active_skill.version}"
            if rows[0].active_skill
            else None,
            rows[0].step_index,
            rows[-1].step_index + 1,
            tuple(rows),
        )
        for rows in groups
    )
