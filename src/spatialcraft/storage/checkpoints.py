"""Immutable, checksummed checkpoints for resumable SpatialCraft runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

from spatialcraft.schemas._base import (
    SchemaMixin,
    new_id,
    normalize_datetime,
    require_non_empty,
    utc_now,
)

from .atomic_io import (
    atomic_copy,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    file_lock,
    read_json,
    sha256_file,
    staged_directory,
)
from .layout import StorageLayout, ensure_within, safe_name, safe_relative_path

CheckpointPayload: TypeAlias = bytes | bytearray | memoryview | Path


class CheckpointError(RuntimeError):
    """Base class for checkpoint storage failures."""


class CheckpointNotFoundError(CheckpointError, FileNotFoundError):
    """Raised when a checkpoint or pointer does not exist."""


class CheckpointConflictError(CheckpointError):
    """Raised when immutable or monotonic checkpoint state would be replaced."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint metadata or payload bytes are invalid."""


def _validate_digest(value: str) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    return digest


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointFile(SchemaMixin):
    """One regular file contained in an immutable checkpoint directory."""

    relative_path: str
    sha256: str
    byte_size: int
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relative = safe_relative_path(self.relative_path, "relative_path").as_posix()
        if relative == "manifest.json":
            raise ValueError("manifest.json is reserved for checkpoint metadata")
        object.__setattr__(self, "relative_path", relative)
        object.__setattr__(self, "sha256", _validate_digest(self.sha256))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.byte_size < 0:
            raise ValueError("byte_size cannot be negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointManifest(SchemaMixin):
    """Self-contained index for one checkpoint generation."""

    run_id: str
    iteration: int
    files: tuple[CheckpointFile, ...]
    checkpoint_id: str = field(default_factory=lambda: new_id("checkpoint"))
    knowledge_snapshot_id: str | None = None
    schema_version: str = "1.0"
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("run_id", "checkpoint_id", "schema_version"):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(getattr(self, field_name), field_name),
            )
        if self.knowledge_snapshot_id is not None:
            object.__setattr__(
                self,
                "knowledge_snapshot_id",
                require_non_empty(self.knowledge_snapshot_id, "knowledge_snapshot_id"),
            )
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.iteration < 0:
            raise ValueError("checkpoint iteration cannot be negative")
        if not self.files:
            raise ValueError("checkpoint must contain at least one file")
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("checkpoint files must have unique relative paths")


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointPointer(SchemaMixin):
    """Small atomic pointer to a checkpoint manifest."""

    checkpoint_id: str
    manifest_uri: str
    iteration: int
    name: str = "latest"
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for field_name in ("checkpoint_id", "manifest_uri", "name"):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))
        if self.iteration < 0:
            raise ValueError("checkpoint pointer iteration cannot be negative")


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except FileNotFoundError as exc:
        raise CheckpointNotFoundError(
            f"checkpoint file does not exist: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(
            f"cannot read checkpoint JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"checkpoint JSON must be an object: {path}")
    return value


class CheckpointStore:
    """Save and resolve immutable multi-file checkpoints.

    This layer intentionally stores bytes and files, not Python pickles. Model code
    may choose its own serialization format, while this store guarantees atomic
    publication, path containment, and byte-level integrity.
    """

    def __init__(self, layout: StorageLayout, *, lock_timeout: float = 60.0) -> None:
        if lock_timeout < 0:
            raise ValueError("lock_timeout cannot be negative")
        self.layout = layout
        self.lock_timeout = float(lock_timeout)

    def save(
        self,
        *,
        run_id: str,
        iteration: int,
        payloads: Mapping[str, CheckpointPayload],
        checkpoint_id: str | None = None,
        knowledge_snapshot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        publish_as: str | None = "latest",
    ) -> CheckpointManifest:
        """Commit payloads as one checkpoint and optionally update a pointer."""

        if iteration < 0:
            raise ValueError("checkpoint iteration cannot be negative")
        if not payloads:
            raise ValueError("payloads cannot be empty")
        run_id = safe_name(run_id, "run_id")
        checkpoint_id = safe_name(
            checkpoint_id or new_id("checkpoint"), "checkpoint_id"
        )
        normalized_payloads: list[tuple[Path, CheckpointPayload]] = []
        seen: set[str] = set()
        for name, payload in payloads.items():
            relative = safe_relative_path(name, "payload name")
            relative_text = relative.as_posix()
            if relative_text == "manifest.json":
                raise ValueError("manifest.json is a reserved payload name")
            if relative_text in seen:
                raise ValueError(f"duplicate checkpoint payload path: {relative_text}")
            if not isinstance(payload, (bytes, bytearray, memoryview, Path)):
                raise TypeError(
                    f"payload {relative_text!r} must be bytes-like or pathlib.Path"
                )
            seen.add(relative_text)
            normalized_payloads.append((relative, payload))
        normalized_payloads.sort(key=lambda item: item[0].as_posix())

        self.layout.ensure_run(run_id)
        target = self.layout.checkpoint_dir(run_id, checkpoint_id)
        lock_path = self.layout.checkpoint_entries_dir(run_id) / ".write.lock"
        with file_lock(lock_path, timeout=self.lock_timeout):
            if target.exists():
                raise CheckpointConflictError(
                    f"checkpoint id already exists: {checkpoint_id}"
                )
            with staged_directory(target) as staging:
                entries: list[CheckpointFile] = []
                for relative, payload in normalized_payloads:
                    destination = ensure_within(
                        staging, staging / relative, "checkpoint payload"
                    )
                    if isinstance(payload, Path):
                        atomic_copy(payload, destination, overwrite=False)
                    else:
                        atomic_write_bytes(destination, payload, overwrite=False)
                    entries.append(
                        CheckpointFile(
                            relative_path=relative.as_posix(),
                            sha256=sha256_file(destination),
                            byte_size=destination.stat().st_size,
                        )
                    )
                manifest = CheckpointManifest(
                    run_id=run_id,
                    iteration=iteration,
                    files=tuple(entries),
                    checkpoint_id=checkpoint_id,
                    knowledge_snapshot_id=knowledge_snapshot_id,
                    metadata=dict(metadata or {}),
                )
                atomic_write_json(
                    staging / "manifest.json", manifest.to_dict(), overwrite=False
                )
        if publish_as is not None:
            self.publish(manifest, name=publish_as)
        return manifest

    def save_json(
        self,
        *,
        run_id: str,
        iteration: int,
        values: Mapping[str, Any],
        checkpoint_id: str | None = None,
        knowledge_snapshot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        publish_as: str | None = "latest",
    ) -> CheckpointManifest:
        """Save named JSON values using deterministic UTF-8 encoding."""

        payloads: dict[str, bytes] = {}
        for name, value in values.items():
            relative = safe_relative_path(name, "JSON payload name")
            if relative.suffix != ".json":
                relative = relative.with_name(relative.name + ".json")
            payloads[relative.as_posix()] = canonical_json_bytes(value)
        return self.save(
            run_id=run_id,
            iteration=iteration,
            payloads=payloads,
            checkpoint_id=checkpoint_id,
            knowledge_snapshot_id=knowledge_snapshot_id,
            metadata=metadata,
            publish_as=publish_as,
        )

    def load(self, run_id: str, checkpoint_id: str) -> CheckpointManifest:
        """Load and strictly reconstruct a checkpoint manifest."""

        path = self.layout.checkpoint_manifest_path(run_id, checkpoint_id)
        try:
            manifest = CheckpointManifest.from_dict(_load_mapping(path))
        except CheckpointError:
            raise
        except (TypeError, ValueError) as exc:
            raise CheckpointIntegrityError(
                f"invalid checkpoint manifest {path}: {exc}"
            ) from exc
        if manifest.run_id != safe_name(run_id, "run_id"):
            raise CheckpointIntegrityError(
                f"checkpoint run_id does not match its namespace: {path}"
            )
        if manifest.checkpoint_id != safe_name(checkpoint_id, "checkpoint_id"):
            raise CheckpointIntegrityError(
                f"checkpoint_id does not match its directory: {path}"
            )
        return manifest

    def verify(
        self,
        manifest: CheckpointManifest,
        *,
        reject_extra_files: bool = True,
    ) -> dict[str, Path]:
        """Verify listed files and optionally reject untracked payloads."""

        root = self.layout.checkpoint_dir(manifest.run_id, manifest.checkpoint_id)
        verified: dict[str, Path] = {}
        for entry in manifest.files:
            relative = safe_relative_path(entry.relative_path)
            path = ensure_within(root, root / relative, "checkpoint file")
            if not path.is_file():
                raise CheckpointIntegrityError(f"missing checkpoint payload: {path}")
            actual_size = path.stat().st_size
            if actual_size != entry.byte_size:
                raise CheckpointIntegrityError(
                    f"size mismatch for {entry.relative_path}: "
                    f"expected {entry.byte_size}, got {actual_size}"
                )
            actual_digest = sha256_file(path)
            if actual_digest != entry.sha256:
                raise CheckpointIntegrityError(
                    f"digest mismatch for {entry.relative_path}: "
                    f"expected {entry.sha256}, got {actual_digest}"
                )
            verified[entry.relative_path] = path

        if reject_extra_files:
            expected = set(verified) | {"manifest.json"}
            actual = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            }
            extra = actual - expected
            if extra:
                raise CheckpointIntegrityError(
                    f"checkpoint contains untracked files: {sorted(extra)}"
                )
        return verified

    def resolve_file(
        self, run_id: str, checkpoint_id: str, relative_path: str | Path
    ) -> Path:
        """Resolve one listed payload after verifying its size and digest."""

        requested = safe_relative_path(relative_path).as_posix()
        manifest = self.load(run_id, checkpoint_id)
        files = self.verify(manifest)
        try:
            return files[requested]
        except KeyError as exc:
            raise CheckpointNotFoundError(
                f"checkpoint {checkpoint_id} has no payload {requested!r}"
            ) from exc

    def publish(
        self,
        manifest: CheckpointManifest,
        *,
        name: str = "latest",
        allow_rewind: bool = False,
    ) -> CheckpointPointer:
        """Atomically move a named pointer to a verified checkpoint."""

        stored = self.load(manifest.run_id, manifest.checkpoint_id)
        if stored.to_dict() != manifest.to_dict():
            raise CheckpointConflictError(
                "checkpoint manifest differs from its persisted representation"
            )
        self.verify(stored)
        pointer_name = safe_name(name, "pointer name")
        pointer_path = self.layout.checkpoint_pointer_path(
            manifest.run_id, pointer_name
        )
        lock_path = pointer_path.with_name(pointer_path.name + ".lock")
        with file_lock(lock_path, timeout=self.lock_timeout):
            if pointer_path.exists():
                current = self.load_pointer(manifest.run_id, pointer_name)
                if current.checkpoint_id == manifest.checkpoint_id:
                    return current
                if not allow_rewind and current.iteration >= manifest.iteration:
                    raise CheckpointConflictError(
                        f"pointer {pointer_name!r} cannot move from iteration "
                        f"{current.iteration} to {manifest.iteration}"
                    )
            pointer = CheckpointPointer(
                checkpoint_id=manifest.checkpoint_id,
                manifest_uri=self.layout.relative_uri(
                    self.layout.checkpoint_manifest_path(
                        manifest.run_id, manifest.checkpoint_id
                    )
                ),
                iteration=manifest.iteration,
                name=pointer_name,
            )
            atomic_write_json(pointer_path, pointer.to_dict())
        return pointer

    def load_pointer(self, run_id: str, name: str = "latest") -> CheckpointPointer:
        """Load a named checkpoint pointer."""

        path = self.layout.checkpoint_pointer_path(run_id, name)
        try:
            pointer = CheckpointPointer.from_dict(_load_mapping(path))
        except CheckpointError:
            raise
        except (TypeError, ValueError) as exc:
            raise CheckpointIntegrityError(
                f"invalid checkpoint pointer {path}: {exc}"
            ) from exc
        if pointer.name != safe_name(name, "pointer name"):
            raise CheckpointIntegrityError(
                f"pointer name does not match its filename: {path}"
            )
        return pointer

    def resolve_pointer(self, run_id: str, name: str = "latest") -> CheckpointManifest:
        """Resolve a pointer and ensure its manifest remains in the run namespace."""

        pointer = self.load_pointer(run_id, name)
        actual_path = self.layout.resolve_uri(pointer.manifest_uri)
        expected_path = self.layout.checkpoint_manifest_path(
            run_id, pointer.checkpoint_id
        ).resolve(strict=False)
        if actual_path != expected_path:
            raise CheckpointIntegrityError(
                f"pointer {name!r} targets a checkpoint outside its run namespace"
            )
        manifest = self.load(run_id, pointer.checkpoint_id)
        if manifest.iteration != pointer.iteration:
            raise CheckpointIntegrityError(
                f"pointer iteration disagrees with checkpoint {pointer.checkpoint_id}"
            )
        return manifest

    def list(self, run_id: str) -> tuple[CheckpointManifest, ...]:
        """List checkpoint manifests in deterministic iteration order."""

        directory = self.layout.checkpoint_entries_dir(run_id)
        if not directory.exists():
            return ()
        manifests = [
            self.load(run_id, path.name)
            for path in directory.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        return tuple(
            sorted(
                manifests,
                key=lambda item: (item.iteration, item.created_at, item.checkpoint_id),
            )
        )


__all__ = [
    "CheckpointConflictError",
    "CheckpointError",
    "CheckpointFile",
    "CheckpointIntegrityError",
    "CheckpointManifest",
    "CheckpointNotFoundError",
    "CheckpointPayload",
    "CheckpointPointer",
    "CheckpointStore",
]
