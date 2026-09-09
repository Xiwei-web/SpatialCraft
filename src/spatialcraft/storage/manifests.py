"""Immutable knowledge manifests and atomic deployment pointers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spatialcraft.schemas import (
    KnowledgeSnapshot,
    SnapshotComponentKind,
    SnapshotComponentRef,
    SnapshotPointer,
)

from .atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    file_lock,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .layout import StorageLayout, safe_name


class ManifestError(RuntimeError):
    """Base class for snapshot manifest failures."""


class ManifestNotFoundError(ManifestError, FileNotFoundError):
    """Raised when a requested manifest or pointer does not exist."""


class ManifestConflictError(ManifestError):
    """Raised when an immutable manifest or monotonic pointer would be replaced."""


class ManifestIntegrityError(ManifestError):
    """Raised when a referenced component fails an integrity check."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except FileNotFoundError as exc:
        raise ManifestNotFoundError(f"manifest file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestIntegrityError(
            f"cannot read JSON manifest {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ManifestIntegrityError(f"manifest must contain a JSON object: {path}")
    return value


class SnapshotManifestStore:
    """Persist and validate :class:`KnowledgeSnapshot` objects.

    Snapshot manifests and content-addressed components are immutable. Only small
    named pointers such as ``latest`` are replaced in place, and those replacements
    use atomic writes guarded by advisory locks.
    """

    def __init__(self, layout: StorageLayout, *, lock_timeout: float = 60.0) -> None:
        if lock_timeout < 0:
            raise ValueError("lock_timeout cannot be negative")
        self.layout = layout
        self.lock_timeout = float(lock_timeout)

    def store_component_bytes(
        self,
        *,
        kind: SnapshotComponentKind,
        payload: bytes | bytearray | memoryview,
        item_count: int,
        suffix: str = ".bin",
        schema_version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotComponentRef:
        """Store component bytes once and return their content-addressed reference."""

        if item_count < 0:
            raise ValueError("item_count cannot be negative")
        data = bytes(payload)
        digest = sha256_bytes(data)
        path = self.layout.object_path(digest, suffix=suffix)
        lock_path = path.with_name(path.name + ".lock")
        with file_lock(lock_path, timeout=self.lock_timeout):
            if path.exists():
                if not path.is_file() or path.stat().st_size != len(data):
                    raise ManifestIntegrityError(
                        f"existing content-addressed object is invalid: {path}"
                    )
                if sha256_file(path) != digest:
                    raise ManifestIntegrityError(
                        f"existing content-addressed object has wrong digest: {path}"
                    )
            else:
                atomic_write_bytes(path, data, overwrite=False)
        return SnapshotComponentRef(
            kind=kind,
            uri=self.layout.relative_uri(path),
            sha256=digest,
            item_count=item_count,
            schema_version=schema_version,
            byte_size=len(data),
            metadata=dict(metadata or {}),
        )

    def store_component_json(
        self,
        *,
        kind: SnapshotComponentKind,
        value: Any,
        item_count: int,
        schema_version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotComponentRef:
        """Store a canonical JSON component in the global object pool."""

        return self.store_component_bytes(
            kind=kind,
            payload=canonical_json_bytes(value),
            item_count=item_count,
            suffix=".json",
            schema_version=schema_version,
            metadata=metadata,
        )

    def component_ref(
        self,
        *,
        kind: SnapshotComponentKind,
        path: str | Path,
        item_count: int,
        schema_version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotComponentRef:
        """Create a checked reference to an existing file below the storage root."""

        component_path = Path(path).expanduser().resolve(strict=False)
        if not component_path.is_file():
            raise FileNotFoundError(
                f"snapshot component is not a file: {component_path}"
            )
        if item_count < 0:
            raise ValueError("item_count cannot be negative")
        return SnapshotComponentRef(
            kind=kind,
            uri=self.layout.relative_uri(component_path),
            sha256=sha256_file(component_path),
            item_count=item_count,
            schema_version=schema_version,
            byte_size=component_path.stat().st_size,
            metadata=dict(metadata or {}),
        )

    def save(
        self,
        snapshot: KnowledgeSnapshot,
        *,
        validate_parent: bool = True,
    ) -> Path:
        """Write one immutable manifest, idempotently for identical content."""

        self.layout.ensure_run(snapshot.run_id)
        target = self.layout.snapshot_manifest_path(
            snapshot.run_id, snapshot.snapshot_id
        )
        lock_path = self.layout.snapshot_manifests_dir(snapshot.run_id) / ".write.lock"
        with file_lock(lock_path, timeout=self.lock_timeout):
            if target.exists():
                existing = self.load(snapshot.run_id, snapshot.snapshot_id)
                if existing.to_dict() != snapshot.to_dict():
                    raise ManifestConflictError(
                        f"snapshot id already contains different content: "
                        f"{snapshot.snapshot_id}"
                    )
                return target
            if validate_parent:
                self._validate_parent(snapshot)
            atomic_write_json(target, snapshot.to_dict(), overwrite=False)
        return target

    def load(self, run_id: str, snapshot_id: str) -> KnowledgeSnapshot:
        """Load and strictly reconstruct a snapshot manifest."""

        path = self.layout.snapshot_manifest_path(run_id, snapshot_id)
        try:
            snapshot = KnowledgeSnapshot.from_dict(_load_mapping(path))
        except ManifestError:
            raise
        except (TypeError, ValueError) as exc:
            raise ManifestIntegrityError(
                f"invalid snapshot manifest {path}: {exc}"
            ) from exc
        if snapshot.run_id != safe_name(run_id, "run_id"):
            raise ManifestIntegrityError(
                f"snapshot run_id does not match its namespace: {path}"
            )
        if snapshot.snapshot_id != safe_name(snapshot_id, "snapshot_id"):
            raise ManifestIntegrityError(
                f"snapshot_id does not match its manifest filename: {path}"
            )
        return snapshot

    def verify(self, snapshot: KnowledgeSnapshot) -> dict[SnapshotComponentKind, Path]:
        """Verify every component and return resolved local paths by kind."""

        verified: dict[SnapshotComponentKind, Path] = {}
        for component in snapshot.components:
            try:
                path = self.layout.resolve_uri(component.uri)
            except ValueError as exc:
                raise ManifestIntegrityError(
                    f"invalid URI for {component.kind.value}: {component.uri}"
                ) from exc
            if not path.is_file():
                raise ManifestIntegrityError(
                    f"missing component {component.kind.value}: {path}"
                )
            actual_size = path.stat().st_size
            if component.byte_size is not None and actual_size != component.byte_size:
                raise ManifestIntegrityError(
                    f"size mismatch for {component.kind.value}: "
                    f"expected {component.byte_size}, got {actual_size}"
                )
            actual_digest = sha256_file(path)
            if actual_digest != component.sha256:
                raise ManifestIntegrityError(
                    f"digest mismatch for {component.kind.value}: "
                    f"expected {component.sha256}, got {actual_digest}"
                )
            verified[component.kind] = path
        return verified

    def publish(
        self,
        snapshot: KnowledgeSnapshot,
        *,
        name: str = "latest",
        verify_components: bool = True,
        allow_rewind: bool = False,
    ) -> SnapshotPointer:
        """Save a snapshot and atomically move a named pointer to it."""

        manifest_path = self.save(snapshot)
        if verify_components:
            self.verify(snapshot)
        pointer_name = safe_name(name, "pointer name")
        pointer_path = self.layout.snapshot_pointer_path(snapshot.run_id, pointer_name)
        lock_path = pointer_path.with_name(pointer_path.name + ".lock")
        with file_lock(lock_path, timeout=self.lock_timeout):
            if pointer_path.exists():
                current_pointer = self.load_pointer(snapshot.run_id, pointer_name)
                if current_pointer.snapshot_id == snapshot.snapshot_id:
                    return current_pointer
                if not allow_rewind:
                    current = self.resolve_pointer(snapshot.run_id, pointer_name)
                    if current.iteration >= snapshot.iteration:
                        raise ManifestConflictError(
                            f"pointer {pointer_name!r} cannot move from iteration "
                            f"{current.iteration} to {snapshot.iteration}"
                        )
            pointer = SnapshotPointer(
                snapshot_id=snapshot.snapshot_id,
                manifest_uri=self.layout.relative_uri(manifest_path),
                name=pointer_name,
            )
            atomic_write_json(pointer_path, pointer.to_dict())
        return pointer

    def load_pointer(self, run_id: str, name: str = "latest") -> SnapshotPointer:
        """Load a named snapshot pointer."""

        path = self.layout.snapshot_pointer_path(run_id, name)
        try:
            pointer = SnapshotPointer.from_dict(_load_mapping(path))
        except ManifestError:
            raise
        except (TypeError, ValueError) as exc:
            raise ManifestIntegrityError(
                f"invalid snapshot pointer {path}: {exc}"
            ) from exc
        if pointer.name != safe_name(name, "pointer name"):
            raise ManifestIntegrityError(
                f"pointer name does not match its filename: {path}"
            )
        return pointer

    def resolve_pointer(self, run_id: str, name: str = "latest") -> KnowledgeSnapshot:
        """Resolve a pointer and ensure it targets the expected run namespace."""

        pointer = self.load_pointer(run_id, name)
        actual_path = self.layout.resolve_uri(pointer.manifest_uri)
        expected_path = self.layout.snapshot_manifest_path(run_id, pointer.snapshot_id)
        if actual_path != expected_path.resolve(strict=False):
            raise ManifestIntegrityError(
                f"pointer {name!r} targets a manifest outside its run namespace"
            )
        return self.load(run_id, pointer.snapshot_id)

    def list(self, run_id: str) -> tuple[KnowledgeSnapshot, ...]:
        """List valid snapshot manifests ordered by iteration and creation time."""

        directory = self.layout.snapshot_manifests_dir(run_id)
        if not directory.exists():
            return ()
        snapshots = [self.load(run_id, path.stem) for path in directory.glob("*.json")]
        return tuple(
            sorted(
                snapshots,
                key=lambda item: (item.iteration, item.created_at, item.snapshot_id),
            )
        )

    def _validate_parent(self, snapshot: KnowledgeSnapshot) -> None:
        if snapshot.parent_snapshot_id is None:
            return
        parent = self.load(snapshot.run_id, snapshot.parent_snapshot_id)
        if snapshot.iteration != parent.iteration + 1:
            raise ManifestConflictError(
                "child snapshot iteration must be exactly parent iteration + 1"
            )
        if snapshot.created_at < parent.created_at:
            raise ManifestConflictError(
                "child snapshot cannot be created before its parent"
            )


__all__ = [
    "ManifestConflictError",
    "ManifestError",
    "ManifestIntegrityError",
    "ManifestNotFoundError",
    "SnapshotManifestStore",
]
