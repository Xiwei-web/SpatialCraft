"""Pure skill lifetime and termination rules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from spatialcraft.schemas import (
    ActionType,
    ActiveSkillRef,
    AgentAction,
    SkillItem,
    SpatialState,
)


@dataclass(frozen=True, slots=True)
class TerminationDecision:
    terminate: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillLifecyclePolicy:
    max_lifetime_steps: int = 8
    condition_evaluator: (
        Callable[[SkillItem, SpatialState, AgentAction], bool] | None
    ) = None

    def __post_init__(self) -> None:
        if self.max_lifetime_steps < 1:
            raise ValueError("max_lifetime_steps must be positive")

    def evaluate(
        self,
        skill: SkillItem,
        active: ActiveSkillRef,
        state: SpatialState,
        action: AgentAction,
    ) -> TerminationDecision:
        if action.action_type is ActionType.FINAL:
            return TerminationDecision(True, "agent_final_answer")
        if active.age_steps + 1 >= self.max_lifetime_steps:
            return TerminationDecision(True, "maximum_lifetime")
        if bool(action.metadata.get("terminate_skill")):
            return TerminationDecision(True, "action_requested")
        if bool(state.metadata.get("skill_done")):
            return TerminationDecision(True, "state_requested")
        if self.condition_evaluator is not None and self.condition_evaluator(
            skill, state, action
        ):
            return TerminationDecision(True, "skill_termination_condition")
        return TerminationDecision(False)


__all__ = ["SkillLifecyclePolicy", "TerminationDecision"]
