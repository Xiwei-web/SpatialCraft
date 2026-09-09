"""Duplicate, low-quality, and stale skill pruning."""

from __future__ import annotations

import re

from .pool import SkillPool


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


class SkillMaintenance:
    def enforce_capacity(
        self, pool: SkillPool, *, capacity: int = 20, duplicate_threshold: float = 0.9
    ):
        """Keep higher online average gain first, then remove redundancy/overflow."""
        if pool.frozen:
            raise ValueError("Capacity maintenance cannot modify deployment knowledge")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if len(pool.active()) <= capacity:
            return pool, ()
        ranked = sorted(
            pool.active(),
            key=lambda item: (
                -item.stats.average_gain,
                -item.stats.frequency,
                item.reference,
            ),
        )
        retained, removed = [], []
        for item in ranked:
            text = " ".join((item.initiation, *item.policy, item.termination))
            redundant = any(
                _similarity(
                    text, " ".join((other.initiation, *other.policy, other.termination))
                )
                >= duplicate_threshold
                for other in retained
            )
            if redundant or len(retained) >= capacity:
                removed.append(item.reference)
            else:
                retained.append(item)
        return self.apply(pool, tuple(removed)), tuple(removed)

    def prune_references(
        self,
        pool: SkillPool,
        *,
        duplicate_threshold: float = 0.9,
        minimum_frequency: int = 3,
        maximum_average_gain: float = 0.0,
        current_iteration: int | None = None,
        stale_after: int = 10,
    ) -> tuple[str, ...]:
        active = list(pool.active())
        prune: set[str] = set()
        for index, left in enumerate(active):
            left_text = " ".join((left.initiation, *left.policy, left.termination))
            for right in active[index + 1 :]:
                right_text = " ".join(
                    (right.initiation, *right.policy, right.termination)
                )
                if _similarity(left_text, right_text) >= duplicate_threshold:
                    loser = min(
                        (left, right),
                        key=lambda item: (
                            item.stats.average_gain,
                            item.stats.frequency,
                            item.version,
                            item.reference,
                        ),
                    )
                    prune.add(loser.reference)
        for item in active:
            if (
                item.stats.frequency >= minimum_frequency
                and item.stats.average_gain <= maximum_average_gain
            ):
                prune.add(item.reference)
            if (
                current_iteration is not None
                and item.stats.last_evolved_iteration is not None
                and current_iteration - item.stats.last_evolved_iteration >= stale_after
                and item.stats.average_gain <= 0.0
            ):
                prune.add(item.reference)
        return tuple(sorted(prune))

    def apply(self, pool: SkillPool, references: tuple[str, ...]) -> SkillPool:
        output = pool
        for reference in references:
            output = output.archive(reference)
        return output


__all__ = ["SkillMaintenance"]
