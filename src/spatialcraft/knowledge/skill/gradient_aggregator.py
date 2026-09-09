"""Aggregate trajectory-specific semantic gradients into candidate evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from spatialcraft.schemas import SemanticGradient

from .semantic_gradient import gradient_reference


def _unique_text(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


@dataclass(frozen=True, slots=True, kw_only=True)
class AggregatedGradient:
    aggregation_id: str
    parent_skill_ref: str | None
    source_gradient_ids: tuple[str, ...]
    source_trajectory_ids: tuple[str, ...]
    initiation_updates: tuple[str, ...]
    policy_updates: tuple[str, ...]
    termination_updates: tuple[str, ...]
    diagnoses: tuple[str, ...]
    mean_reward: float

    @property
    def support(self) -> int:
        return len(self.source_gradient_ids)


class GradientAggregator:
    def aggregate(
        self, gradients: tuple[SemanticGradient, ...]
    ) -> tuple[AggregatedGradient, ...]:
        groups: dict[str | None, list[SemanticGradient]] = {}
        for gradient in gradients:
            references = tuple(gradient.metadata.get("skill_references", ()))
            keys: tuple[str | None, ...] = (
                references if gradient.is_related else (None,)
            )
            for key in keys:
                groups.setdefault(key, []).append(gradient)
        output = []
        for parent, values in sorted(groups.items(), key=lambda item: item[0] or ""):
            source_ids = tuple(gradient_reference(value) for value in values)
            digest = sha256(f"{parent}|{'|'.join(source_ids)}".encode()).hexdigest()[
                :20
            ]
            rewards = [float(value.reward or 0.0) for value in values]
            output.append(
                AggregatedGradient(
                    aggregation_id=f"gradient_aggregate_{digest}",
                    parent_skill_ref=parent,
                    source_gradient_ids=source_ids,
                    source_trajectory_ids=tuple(
                        dict.fromkeys(value.trajectory_id for value in values)
                    ),
                    initiation_updates=_unique_text(
                        [value.initiation for value in values]
                    ),
                    policy_updates=_unique_text([value.policy for value in values]),
                    termination_updates=_unique_text(
                        [value.termination for value in values]
                    ),
                    diagnoses=_unique_text([value.diagnosis for value in values]),
                    mean_reward=sum(rewards) / len(rewards),
                )
            )
        return tuple(output)


__all__ = ["AggregatedGradient", "GradientAggregator"]
