"""Canonical SpatialCraft ablations and parameter/model sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True, kw_only=True)
class AblationSpec:
    name: str
    overrides: dict = field(default_factory=dict)


def standard_ablations() -> tuple[AblationSpec, ...]:
    return (
        AblationSpec(name="no_experience", overrides={"experience.enabled": False}),
        AblationSpec(name="no_skill", overrides={"skill.enabled": False}),
        AblationSpec(
            name="no_task_decomposition",
            overrides={"experience.task_decomposition": False},
        ),
        AblationSpec(
            name="no_experience_rewrite", overrides={"experience.rewrite": False}
        ),
        AblationSpec(
            name="no_visual_summary", overrides={"experience.visual_summary": False}
        ),
        AblationSpec(
            name="no_cross_rollout_critique",
            overrides={"experience.cross_rollout_critique": False},
        ),
        AblationSpec(
            name="no_semantic_gradient", overrides={"skill.semantic_gradient": False}
        ),
        AblationSpec(name="no_ppo_gate", overrides={"skill.ppo_gate": False}),
        AblationSpec(
            name="no_skill_score_pruning", overrides={"skill.score_pruning": False}
        ),
        AblationSpec(
            name="static_seed_skills",
            overrides={"skill.learning": False, "skill.seed_only": True},
        ),
    )


def pool_size_ablations(
    experience_sizes: tuple[int, ...], skill_sizes: tuple[int, ...]
) -> tuple[AblationSpec, ...]:
    return tuple(
        AblationSpec(
            name=f"pool_size_e{experience}_s{skill}",
            overrides={
                "experience.max_pool_size": experience,
                "skill.max_pool_size": skill,
            },
        )
        for experience in experience_sizes
        for skill in skill_sizes
    )


def rollout_count_ablations(counts: tuple[int, ...]) -> tuple[AblationSpec, ...]:
    if any(count < 1 for count in counts):
        raise ValueError("rollout counts must be positive")
    return tuple(
        AblationSpec(name=f"rollouts_{count}", overrides={"rollout.count": count})
        for count in counts
    )


def model_pair_ablations(
    executor_models: tuple[str, ...], knowledge_builder_models: tuple[str, ...]
) -> tuple[AblationSpec, ...]:
    return tuple(
        AblationSpec(
            name=f"models_{executor}__{builder}",
            overrides={
                "models.executor": executor,
                "models.knowledge_builder": builder,
            },
        )
        for executor in executor_models
        for builder in knowledge_builder_models
    )


__all__ = [
    "AblationSpec",
    "model_pair_ablations",
    "pool_size_ablations",
    "rollout_count_ablations",
    "standard_ablations",
]
