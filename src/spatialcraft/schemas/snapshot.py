"""Versioned, immutable knowledge-snapshot manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ._base import SchemaMixin, new_id, normalize_datetime, require_non_empty, utc_now


class SnapshotStage(str, Enum):
    INITIAL = "initial"
    ACCUMULATION = "accumulation"
    IMPORTED = "imported"


class SnapshotComponentKind(str, Enum):
    EXPERIENCE_BANK = "experience_bank"
    EXPERIENCE_INDEX = "experience_index"
    SKILL_POOL = "skill_pool"
    SKILL_INDEX = "skill_index"
    CONFIG = "config"
    PROMPTS = "prompts"


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotComponentRef(SchemaMixin):
    """Content-addressed reference to one persisted snapshot component."""

    kind: SnapshotComponentKind
    uri: str
    sha256: str
    item_count: int
    schema_version: str = "1.0"
    byte_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", require_non_empty(self.uri, "uri"))
        object.__setattr__(
            self,
            "schema_version",
            require_non_empty(self.schema_version, "schema_version"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        digest = self.sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "sha256", digest)
        if self.item_count < 0:
            raise ValueError("item_count cannot be negative")
        if self.byte_size is not None and self.byte_size < 0:
            raise ValueError("byte_size cannot be negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeSnapshot(SchemaMixin):
    """Atomic manifest binding Experience and Skill knowledge versions.

    Deployment loads one manifest and never mutates its referenced components.
    A new accumulation update always creates a new snapshot with a parent link.
    """

    run_id: str
    iteration: int
    stage: SnapshotStage
    components: tuple[SnapshotComponentRef, ...]
    snapshot_id: str = field(default_factory=lambda: new_id("snapshot"))
    parent_snapshot_id: str | None = None
    config_digest: str | None = None
    source_trajectory_ids: tuple[str, ...] = ()
    frozen: bool = True
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", require_non_empty(self.run_id, "run_id"))
        object.__setattr__(
            self, "snapshot_id", require_non_empty(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(
            self, "source_trajectory_ids", tuple(self.source_trajectory_ids)
        )
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.iteration < 0:
            raise ValueError("snapshot iteration cannot be negative")
        if not self.frozen:
            raise ValueError("published knowledge snapshots must be immutable")
        if self.parent_snapshot_id == self.snapshot_id:
            raise ValueError("a snapshot cannot be its own parent")
        if self.iteration == 0 and self.stage is not SnapshotStage.INITIAL:
            raise ValueError("iteration zero must use the INITIAL stage")
        if self.iteration > 0 and self.stage is SnapshotStage.INITIAL:
            raise ValueError("INITIAL stage is reserved for iteration zero")
        if self.iteration > 0 and not self.parent_snapshot_id:
            raise ValueError("non-initial snapshots require parent_snapshot_id")

        kinds = [component.kind for component in self.components]
        if len(kinds) != len(set(kinds)):
            raise ValueError("a snapshot cannot contain duplicate component kinds")
        required = {
            SnapshotComponentKind.EXPERIENCE_BANK,
            SnapshotComponentKind.SKILL_POOL,
        }
        missing = required - set(kinds)
        if missing:
            raise ValueError(
                f"snapshot is missing required components: "
                f"{sorted(kind.value for kind in missing)}"
            )
        if len(self.source_trajectory_ids) != len(set(self.source_trajectory_ids)):
            raise ValueError("source_trajectory_ids cannot contain duplicates")
        if self.config_digest is not None:
            digest = self.config_digest.lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(
                    "config_digest must be a 64-character hexadecimal digest"
                )
            object.__setattr__(self, "config_digest", digest)

    def component(self, kind: SnapshotComponentKind) -> SnapshotComponentRef | None:
        """Return a component by kind without exposing mutable snapshot state."""

        return next((item for item in self.components if item.kind is kind), None)


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotPointer(SchemaMixin):
    """Small atomic pointer used for ``latest`` or named deployment releases."""

    snapshot_id: str
    manifest_uri: str
    name: str = "latest"
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for field_name in ("snapshot_id", "manifest_uri", "name"):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))
