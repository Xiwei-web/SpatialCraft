"""Persist one selected skill across agent steps and retire it safely."""

from __future__ import annotations

from dataclasses import replace

from spatialcraft.knowledge.skill import SkillPool, SkillSelector
from spatialcraft.schemas import ActiveSkillRef, AgentAction, SpatialState, TaskSample

from .termination_controller import TerminationController


class SkillController:
    def __init__(
        self,
        pool: SkillPool,
        *,
        selector: SkillSelector | None = None,
        termination: TerminationController | None = None,
        enabled: bool = True,
    ) -> None:
        self.pool = pool.freeze()
        self.selector = selector or SkillSelector(enabled=enabled)
        self.termination = termination or TerminationController()
        self.enabled = enabled

    def before_step(self, task: TaskSample, state: SpatialState) -> SpatialState:
        if not self.enabled or state.active_skill is not None:
            return state
        skill = self.selector.select(self.pool, task, state)
        if skill is None:
            return state
        metadata = dict(state.metadata)
        metadata["active_skill_prompt"] = skill.format_for_prompt()
        metadata["selected_skill_ref"] = skill.reference
        return replace(
            state,
            active_skill=ActiveSkillRef(
                skill_id=skill.skill_id,
                version=skill.version,
                activated_at_step=state.step_index,
            ),
            metadata=metadata,
        )

    def after_step(
        self,
        task: TaskSample,
        state: SpatialState,
        action: AgentAction,
        next_state: SpatialState,
    ) -> SpatialState:
        del task
        if not self.enabled or state.active_skill is None:
            return next_state
        active = state.active_skill
        skill = self.pool.get(f"{active.skill_id}@{active.version}")
        decision = self.termination.evaluate(skill, state, action, next_state)
        metadata = dict(next_state.metadata)
        if decision.terminate:
            retired = tuple(metadata.get("retired_skill_refs", ()))
            reference = skill.reference
            metadata["retired_skill_refs"] = (*retired, reference)
            metadata["last_terminated_skill_ref"] = reference
            metadata["last_skill_termination_reason"] = decision.reason
            metadata.pop("active_skill_prompt", None)
            return replace(next_state, active_skill=None, metadata=metadata)
        return replace(
            next_state,
            active_skill=replace(active, age_steps=active.age_steps + 1),
            metadata=metadata,
        )


__all__ = ["SkillController"]
