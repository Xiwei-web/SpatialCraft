"""Agent-facing wrapper around pure skill termination policy."""

from __future__ import annotations

from spatialcraft.knowledge.skill import SkillLifecyclePolicy, TerminationDecision
from spatialcraft.schemas import AgentAction, SkillItem, SpatialState


class TerminationController:
    def __init__(self, policy: SkillLifecyclePolicy | None = None) -> None:
        self.policy = policy or SkillLifecyclePolicy()

    def evaluate(
        self,
        skill: SkillItem,
        state: SpatialState,
        action: AgentAction,
        next_state: SpatialState | None = None,
    ) -> TerminationDecision:
        if state.active_skill is None:
            return TerminationDecision(False)
        return self.policy.evaluate(
            skill, state.active_skill, next_state or state, action
        )


__all__ = ["TerminationController"]
