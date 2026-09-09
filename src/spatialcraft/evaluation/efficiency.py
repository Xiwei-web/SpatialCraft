"""Token/tool/cost efficiency summaries."""

from __future__ import annotations

from dataclasses import dataclass

from spatialcraft.schemas import Trajectory


@dataclass(frozen=True, slots=True, kw_only=True)
class EfficiencyMetrics:
    trajectory_count: int
    mean_steps: float
    mean_tool_calls: float
    mean_state_tokens: float
    mean_cost: float
    reward_per_tool_call: float
    reward_per_1k_tokens: float
    mean_environment_steps: float = 0.0
    mean_normal_llm_calls: float = 0.0
    mean_recovery_calls: float = 0.0
    mean_forced_final_answers: float = 0.0
    mean_token_truncations: float = 0.0


def efficiency(trajectories: tuple[Trajectory, ...]) -> EfficiencyMetrics:
    count = len(trajectories)
    steps = sum(len(item.transitions) for item in trajectories)
    calls = sum(item.total_tool_calls for item in trajectories)
    tokens = sum(
        item.transitions[-1].state_after.token_count
        for item in trajectories
        if item.transitions and item.transitions[-1].state_after is not None
    )
    costs = sum(float(item.metadata.get("cost", 0.0)) for item in trajectories)
    reward = sum(float(item.reward or 0.0) for item in trajectories)

    def mean_count(name):
        return (
            sum(
                item.metadata.get("call_counts", {}).get(name, 0)
                for item in trajectories
            )
            / count
            if count
            else 0.0
        )

    return EfficiencyMetrics(
        trajectory_count=count,
        mean_steps=steps / count if count else 0.0,
        mean_tool_calls=calls / count if count else 0.0,
        mean_state_tokens=tokens / count if count else 0.0,
        mean_cost=costs / count if count else 0.0,
        reward_per_tool_call=reward / calls if calls else 0.0,
        reward_per_1k_tokens=reward * 1000 / tokens if tokens else 0.0,
        mean_environment_steps=sum(
            item.metadata.get("environment_steps", item.total_tool_calls)
            for item in trajectories
        )
        / count
        if count
        else 0.0,
        mean_normal_llm_calls=mean_count("normal_llm_calls"),
        mean_recovery_calls=mean_count("recovery_calls"),
        mean_forced_final_answers=mean_count("forced_final_answers"),
        mean_token_truncations=mean_count("token_truncations"),
    )


__all__ = ["EfficiencyMetrics", "efficiency"]
