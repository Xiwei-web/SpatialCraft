"""Generate traceable REFINE and NEW skill candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from spatialcraft.schemas import (
    SkillCandidate,
    SkillEvolutionType,
    SkillItem,
    SkillStats,
)

from .gradient_aggregator import AggregatedGradient
from .pool import SkillPool


def _latest(values: tuple[str, ...], fallback: str) -> str:
    return values[-1] if values else fallback


class SkillCandidateGenerator:
    def __init__(self, generator: Callable | None = None):
        self.generator = generator

    def generate(
        self,
        aggregates: tuple[AggregatedGradient, ...],
        pool: SkillPool,
        *,
        num_candidates: int = 1,
    ) -> tuple[SkillCandidate, ...]:
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        output = []
        for aggregate in aggregates:
            if aggregate.parent_skill_ref is not None:
                parent = pool.get(aggregate.parent_skill_ref)
                policy = tuple(
                    dict.fromkeys((*parent.policy, *aggregate.policy_updates))
                )
                skill = replace(
                    parent,
                    version=parent.version + 1,
                    initiation=_latest(aggregate.initiation_updates, parent.initiation),
                    policy=policy,
                    termination=_latest(
                        aggregate.termination_updates, parent.termination
                    ),
                    evolution_type=SkillEvolutionType.REFINE,
                    parent_skill_ref=parent.reference,
                    stats=SkillStats(),
                    metadata={
                        **parent.metadata,
                        "source_aggregation_id": aggregate.aggregation_id,
                    },
                )
                evolution_type = SkillEvolutionType.REFINE
            else:
                suffix = aggregate.aggregation_id.rsplit("_", 1)[-1][:8]
                skill = SkillItem(
                    name=f"LearnedSpatialSkill_{suffix}",
                    initiation=_latest(
                        aggregate.initiation_updates,
                        "Activate when existing skills do not cover the spatial task.",
                    ),
                    policy=aggregate.policy_updates
                    or ("Gather decisive spatial evidence and verify the conclusion.",),
                    termination=_latest(
                        aggregate.termination_updates,
                        "Terminate when the queried relation is supported.",
                    ),
                    evolution_type=SkillEvolutionType.NEW,
                    metadata={"source_aggregation_id": aggregate.aggregation_id},
                )
                evolution_type = SkillEvolutionType.NEW
            for index in range(num_candidates):
                proposed = skill
                if self.generator is not None:
                    values = self.generator(
                        aggregate,
                        pool.get(aggregate.parent_skill_ref)
                        if aggregate.parent_skill_ref
                        else None,
                        index,
                    )
                    required = ("name", "initiation", "policy", "termination")
                    if any(key not in values for key in required) or not isinstance(
                        values["policy"], (list, tuple)
                    ):
                        raise ValueError(
                            "Candidate generator must return name/initiation/policy/termination"
                        )
                    proposed = replace(skill, **{key: values[key] for key in required})
                output.append(
                    SkillCandidate(
                        skill=proposed,
                        evolution_type=evolution_type,
                        source_gradient_ids=aggregate.source_gradient_ids,
                        source_trajectory_ids=aggregate.source_trajectory_ids,
                        metadata={
                            "aggregation_id": aggregate.aggregation_id,
                            "support": aggregate.support,
                            "mean_reward": aggregate.mean_reward,
                            "candidate_index": index,
                            "candidate_count": num_candidates,
                        },
                    )
                )
        return tuple(output)


__all__ = ["SkillCandidateGenerator"]
