"""Action-level experience memory schemas."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from hashlib import sha256
from math import isfinite
from typing import Any

from ._base import SchemaMixin, new_id, normalize_datetime, require_non_empty, utc_now


class ExperienceStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class ExperienceOperationType(str, Enum):
    ADD = "add"
    MODIFY = "modify"
    MERGE = "merge"
    ARCHIVE = "archive"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperienceProvenance(SchemaMixin):
    """Auditable origins of an experience without embedding full trajectories."""

    trajectory_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    rollout_indices: tuple[int, ...] = ()
    operation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory_ids", tuple(self.trajectory_ids))
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "rollout_indices", tuple(self.rollout_indices))
        if any(index < 0 for index in self.rollout_indices):
            raise ValueError("rollout indices cannot be negative")
        for label, values in (
            ("trajectory_ids", self.trajectory_ids),
            ("task_ids", self.task_ids),
            ("datasets", self.datasets),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} cannot contain duplicates")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperienceStats(SchemaMixin):
    """Usage statistics that do not alter the semantic experience content."""

    retrieval_count: int = 0
    use_count: int = 0
    successful_use_count: int = 0
    cumulative_reward_gain: float = 0.0
    last_used_at: datetime | None = None

    def __post_init__(self) -> None:
        if min(self.retrieval_count, self.use_count, self.successful_use_count) < 0:
            raise ValueError("experience counters cannot be negative")
        if self.successful_use_count > self.use_count:
            raise ValueError("successful_use_count cannot exceed use_count")
        if not isfinite(float(self.cumulative_reward_gain)):
            raise ValueError("cumulative_reward_gain must be finite")
        if self.last_used_at is not None:
            object.__setattr__(
                self, "last_used_at", normalize_datetime(self.last_used_at)
            )

    @property
    def average_reward_gain(self) -> float:
        return self.cumulative_reward_gain / max(1, self.use_count)

    def record_retrieval(self, count: int = 1) -> ExperienceStats:
        if count < 1:
            raise ValueError("retrieval count increment must be positive")
        return replace(self, retrieval_count=self.retrieval_count + count)

    def record_use(
        self, *, reward_gain: float, successful: bool, used_at: datetime | None = None
    ) -> ExperienceStats:
        if not isfinite(float(reward_gain)):
            raise ValueError("reward_gain must be finite")
        return replace(
            self,
            use_count=self.use_count + 1,
            successful_use_count=self.successful_use_count + int(successful),
            cumulative_reward_gain=self.cumulative_reward_gain + float(reward_gain),
            last_used_at=used_at or utc_now(),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperienceItem(SchemaMixin):
    """A reusable condition-action lesson distilled from agent trajectories."""

    condition: str
    action: str
    experience_id: str = field(default_factory=lambda: new_id("experience"))
    version: int = 1
    status: ExperienceStatus = ExperienceStatus.ACTIVE
    summary: str | None = None
    source_experience_refs: tuple[str, ...] = ()
    provenance: ExperienceProvenance = field(default_factory=ExperienceProvenance)
    stats: ExperienceStats = field(default_factory=ExperienceStats)
    embedding_key: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("condition", "action", "experience_id"):
            object.__setattr__(self, name, require_non_empty(getattr(self, name), name))
        object.__setattr__(
            self, "source_experience_refs", tuple(self.source_experience_refs)
        )
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.version < 1:
            raise ValueError("experience version must be at least 1")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        if len(self.source_experience_refs) != len(set(self.source_experience_refs)):
            raise ValueError("source_experience_refs cannot contain duplicates")

    @property
    def reference(self) -> str:
        return f"{self.experience_id}@{self.version}"

    @property
    def prompt_text(self) -> str:
        return f"Condition: {self.condition}\nAction: {self.action}"

    @property
    def content_hash(self) -> str:
        return sha256(self.prompt_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperienceUpdate(SchemaMixin):
    """Structured add/modify/merge/archive proposal from cross-rollout critique."""

    operation: ExperienceOperationType
    rationale: str
    update_id: str = field(default_factory=lambda: new_id("experience_update"))
    target_experience_refs: tuple[str, ...] = ()
    proposed_experience: ExperienceItem | None = None
    source_trajectory_ids: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rationale", require_non_empty(self.rationale, "rationale")
        )
        object.__setattr__(
            self, "update_id", require_non_empty(self.update_id, "update_id")
        )
        object.__setattr__(
            self, "target_experience_refs", tuple(self.target_experience_refs)
        )
        object.__setattr__(
            self, "source_trajectory_ids", tuple(self.source_trajectory_ids)
        )
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "metadata", dict(self.metadata))

        targets = self.target_experience_refs
        if len(targets) != len(set(targets)):
            raise ValueError("target_experience_refs cannot contain duplicates")
        if self.operation is ExperienceOperationType.ADD:
            if targets or self.proposed_experience is None:
                raise ValueError("add requires a proposal and no target experiences")
        elif self.operation is ExperienceOperationType.MODIFY:
            if len(targets) != 1 or self.proposed_experience is None:
                raise ValueError("modify requires one target and one proposal")
        elif self.operation is ExperienceOperationType.MERGE:
            if len(targets) < 2 or self.proposed_experience is None:
                raise ValueError("merge requires at least two targets and one proposal")
        elif self.operation is ExperienceOperationType.ARCHIVE and (
            not targets or self.proposed_experience is not None
        ):
            raise ValueError("archive requires targets and no proposal")
