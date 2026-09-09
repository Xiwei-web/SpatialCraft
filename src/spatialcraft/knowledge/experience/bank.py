"""Versioned functional Experience Bank with strict update semantics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from spatialcraft.schemas import (
    ExperienceItem,
    ExperienceOperationType,
    ExperienceStatus,
    ExperienceUpdate,
)


class ExperienceBankError(RuntimeError):
    pass


class ExperienceBank:
    """An immutable-by-default view over all auditable experience versions."""

    def __init__(
        self, items: tuple[ExperienceItem, ...] = (), *, frozen: bool = False
    ) -> None:
        references = [item.reference for item in items]
        if len(references) != len(set(references)):
            raise ValueError("experience references must be unique")
        self._items = {item.reference: item for item in items}
        self.frozen = frozen

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> tuple[ExperienceItem, ...]:
        return tuple(
            sorted(
                self._items.values(),
                key=lambda item: (item.experience_id, item.version),
            )
        )

    def active(self) -> tuple[ExperienceItem, ...]:
        latest: dict[str, ExperienceItem] = {}
        for item in self._items.values():
            if item.status is ExperienceStatus.ACTIVE and (
                item.experience_id not in latest
                or latest[item.experience_id].version < item.version
            ):
                latest[item.experience_id] = item
        return tuple(sorted(latest.values(), key=lambda item: item.experience_id))

    def get(self, reference: str) -> ExperienceItem:
        try:
            return self._items[reference]
        except KeyError as exc:
            raise ExperienceBankError(
                f"unknown experience reference: {reference}"
            ) from exc

    def latest(self, experience_id: str) -> ExperienceItem:
        candidates = [
            item for item in self._items.values() if item.experience_id == experience_id
        ]
        if not candidates:
            raise ExperienceBankError(f"unknown experience id: {experience_id}")
        return max(candidates, key=lambda item: item.version)

    def freeze(self) -> ExperienceBank:
        return ExperienceBank(self.all(), frozen=True)

    def apply(self, update: ExperienceUpdate) -> ExperienceBank:
        if self.frozen:
            raise ExperienceBankError("cannot mutate a frozen Experience Bank")
        values = dict(self._items)
        operation = update.operation
        if operation is ExperienceOperationType.ADD:
            proposal = update.proposed_experience
            assert proposal is not None
            if any(
                item.experience_id == proposal.experience_id for item in values.values()
            ):
                raise ExperienceBankError(
                    f"experience id already exists: {proposal.experience_id}"
                )
            values[proposal.reference] = proposal
        elif operation is ExperienceOperationType.MODIFY:
            target = self.get(update.target_experience_refs[0])
            proposal = update.proposed_experience
            assert proposal is not None
            if proposal.experience_id != target.experience_id:
                raise ExperienceBankError(
                    "modified experience must preserve experience_id"
                )
            if proposal.version != target.version + 1:
                raise ExperienceBankError(
                    "modified experience version must increment by one"
                )
            values[target.reference] = replace(
                target, status=ExperienceStatus.SUPERSEDED
            )
            values[proposal.reference] = proposal
        elif operation is ExperienceOperationType.MERGE:
            targets = tuple(self.get(ref) for ref in update.target_experience_refs)
            proposal = update.proposed_experience
            assert proposal is not None
            if any(
                item.experience_id == proposal.experience_id for item in values.values()
            ):
                raise ExperienceBankError(
                    "merged experience must use a new experience_id"
                )
            for target in targets:
                values[target.reference] = replace(
                    target, status=ExperienceStatus.SUPERSEDED
                )
            values[proposal.reference] = proposal
        elif operation is ExperienceOperationType.ARCHIVE:
            for reference in update.target_experience_refs:
                target = self.get(reference)
                values[target.reference] = replace(
                    target, status=ExperienceStatus.ARCHIVED
                )
        return ExperienceBank(tuple(values.values()))

    def apply_many(self, updates: tuple[ExperienceUpdate, ...]) -> ExperienceBank:
        bank = self
        for update in updates:
            bank = bank.apply(update)
        return bank

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "items": [item.to_dict() for item in self.all()],
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, frozen: bool = False
    ) -> ExperienceBank:
        if value.get("schema_version") != "1.0":
            raise ValueError("unsupported Experience Bank schema version")
        return cls(
            tuple(ExperienceItem.from_dict(item) for item in value.get("items", ())),
            frozen=frozen,
        )


__all__ = ["ExperienceBank", "ExperienceBankError"]
