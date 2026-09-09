"""Versioned, snapshot-friendly procedural skill pool."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from spatialcraft.schemas import SkillItem, SkillStatus


class SkillPoolError(RuntimeError):
    """Raised when a skill-pool invariant is violated."""


class SkillPool:
    """Keep every skill version while exposing only latest active versions."""

    def __init__(
        self, items: tuple[SkillItem, ...] = (), *, frozen: bool = False
    ) -> None:
        references = [item.reference for item in items]
        if len(references) != len(set(references)):
            raise ValueError("skill references must be unique")
        self._items = {item.reference: item for item in items}
        self.frozen = frozen

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> tuple[SkillItem, ...]:
        return tuple(
            sorted(self._items.values(), key=lambda item: (item.skill_id, item.version))
        )

    def active(self) -> tuple[SkillItem, ...]:
        latest: dict[str, SkillItem] = {}
        for item in self._items.values():
            if item.status is SkillStatus.ACTIVE and (
                item.skill_id not in latest
                or latest[item.skill_id].version < item.version
            ):
                latest[item.skill_id] = item
        return tuple(sorted(latest.values(), key=lambda item: item.skill_id))

    def get(self, reference: str) -> SkillItem:
        try:
            return self._items[reference]
        except KeyError as exc:
            raise SkillPoolError(f"unknown skill reference: {reference}") from exc

    def latest(self, skill_id: str) -> SkillItem:
        values = [item for item in self._items.values() if item.skill_id == skill_id]
        if not values:
            raise SkillPoolError(f"unknown skill id: {skill_id}")
        return max(values, key=lambda item: item.version)

    def freeze(self) -> SkillPool:
        return SkillPool(self.all(), frozen=True)

    def add(self, skill: SkillItem) -> SkillPool:
        if self.frozen:
            raise SkillPoolError("cannot mutate a frozen Skill Pool")
        if skill.reference in self._items:
            raise SkillPoolError(f"skill reference already exists: {skill.reference}")
        if any(item.skill_id == skill.skill_id for item in self._items.values()):
            raise SkillPoolError(
                "add requires a new skill_id; use refine for a new version"
            )
        return SkillPool((*self.all(), skill))

    def refine(self, skill: SkillItem) -> SkillPool:
        if self.frozen:
            raise SkillPoolError("cannot mutate a frozen Skill Pool")
        parent = self.latest(skill.skill_id)
        if skill.version != parent.version + 1:
            raise SkillPoolError("refined skill version must increment by one")
        if skill.parent_skill_ref != parent.reference:
            raise SkillPoolError("refined skill must reference its latest parent")
        values = dict(self._items)
        values[parent.reference] = replace(parent, status=SkillStatus.SUPERSEDED)
        values[skill.reference] = skill
        return SkillPool(tuple(values.values()))

    def archive(self, reference: str) -> SkillPool:
        if self.frozen:
            raise SkillPoolError("cannot mutate a frozen Skill Pool")
        target = self.get(reference)
        values = dict(self._items)
        values[reference] = replace(target, status=SkillStatus.ARCHIVED)
        return SkillPool(tuple(values.values()))

    def replace_item(self, skill: SkillItem) -> SkillPool:
        """Replace one exact version, primarily for immutable statistics updates."""

        if self.frozen:
            raise SkillPoolError("cannot mutate a frozen Skill Pool")
        if skill.reference not in self._items:
            raise SkillPoolError(f"unknown skill reference: {skill.reference}")
        values = dict(self._items)
        values[skill.reference] = skill
        return SkillPool(tuple(values.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "items": [item.to_dict() for item in self.all()],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, frozen: bool = False) -> SkillPool:
        if value.get("schema_version") != "1.0":
            raise ValueError("unsupported Skill Pool schema version")
        return cls(
            tuple(SkillItem.from_dict(item) for item in value.get("items", ())),
            frozen=frozen,
        )


__all__ = ["SkillPool", "SkillPoolError"]
