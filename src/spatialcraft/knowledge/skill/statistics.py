"""Functional updates for per-version skill utility statistics."""

from __future__ import annotations

from dataclasses import replace

from .credit_assignment import SkillCredit
from .pool import SkillPool


class SkillStatistics:
    def update(
        self,
        pool: SkillPool,
        credits: tuple[SkillCredit, ...],
        *,
        iteration: int | None = None,
    ) -> SkillPool:
        if pool.frozen:
            raise ValueError("statistics updates require an unfrozen Skill Pool")
        grouped: dict[str, list[SkillCredit]] = {}
        for credit in credits:
            grouped.setdefault(credit.skill_reference, []).append(credit)
        output = pool
        for reference, values in sorted(grouped.items()):
            item = output.get(reference)
            stats = item.stats
            for credit in values:
                stats = stats.record_usage(
                    advantage=credit.advantage,
                    call_count=1,
                    successful=credit.successful,
                )
            stats = replace(
                stats,
                maturity=stats.maturity + 1,
            )
            output = output.replace_item(replace(item, stats=stats))
        return output


__all__ = ["SkillStatistics"]
