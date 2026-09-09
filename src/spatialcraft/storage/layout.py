"""Deterministic and traversal-safe filesystem layout for SpatialCraft data."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

STORAGE_ROOT_ENV = "SPATIALCRAFT_STORAGE_ROOT"


class InvalidStoragePath(ValueError):
    """Raised when a storage identifier or URI could escape its namespace."""


def safe_name(value: str, field_name: str = "name") -> str:
    """Validate one portable path component.

    Identifiers are deliberately allowed to contain dots, dashes, and underscores,
    but never separators or the special ``.`` and ``..`` components.
    """

    if not isinstance(value, str) or not value.strip():
        raise InvalidStoragePath(f"{field_name} must be a non-empty string")
    value = value.strip()
    if value in {".", ".."}:
        raise InvalidStoragePath(f"{field_name} cannot be {value!r}")
    if "\x00" in value or "/" in value or "\\" in value:
        raise InvalidStoragePath(f"{field_name} must be a single path component")
    if len(value.encode("utf-8")) > 255:
        raise InvalidStoragePath(f"{field_name} is too long for a path component")
    return value


def safe_relative_path(value: str | Path, field_name: str = "relative_path") -> Path:
    """Return a normalized relative path without traversal components."""

    raw = str(value)
    if not raw or "\x00" in raw or "\\" in raw:
        raise InvalidStoragePath(f"{field_name} must be a non-empty POSIX path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or not pure.parts:
        raise InvalidStoragePath(f"{field_name} must be relative")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise InvalidStoragePath(f"{field_name} contains a traversal component")
    for part in pure.parts:
        safe_name(part, field_name)
    return Path(*pure.parts)


def ensure_within(base: Path, candidate: Path, field_name: str = "path") -> Path:
    """Resolve *candidate* and ensure it remains inside *base*."""

    resolved_base = base.expanduser().resolve(strict=False)
    resolved = candidate.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise InvalidStoragePath(
            f"{field_name} escapes storage root {resolved_base}: {candidate}"
        ) from exc
    return resolved


def _validate_suffix(suffix: str) -> str:
    if suffix == "":
        return suffix
    if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
        raise InvalidStoragePath("suffix must be empty or start with '.'")
    safe_name(suffix, "suffix")
    return suffix


def _validate_sha256(digest: str) -> str:
    digest = digest.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise InvalidStoragePath("digest must be a 64-character SHA-256 hex string")
    return digest


@dataclass(frozen=True, slots=True)
class StorageLayout:
    """All durable paths used by a SpatialCraft experiment.

    The root contains global content-addressed objects and isolated run trees::

        root/
          objects/sha256/ab/<digest>.<ext>
          runs/<run_id>/
            artifacts/
            trajectories/
            knowledge/{snapshots,pointers}/
            checkpoints/{entries,pointers}/
            logs/
          tmp/

    Path constructors do not create directories. Call :meth:`ensure` or
    :meth:`ensure_run` at a process boundary that owns initialization.
    """

    root: Path

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve(strict=False)
        object.__setattr__(self, "root", root)

    @classmethod
    def from_env(
        cls,
        root: str | Path | None = None,
        *,
        env_var: str = STORAGE_ROOT_ENV,
    ) -> StorageLayout:
        """Create a layout from an explicit root or ``SPATIALCRAFT_STORAGE_ROOT``."""

        selected = root if root is not None else os.environ.get(env_var)
        if selected is None:
            raise ValueError(f"storage root is required; pass root or set {env_var}")
        return cls(Path(selected))

    @property
    def objects_dir(self) -> Path:
        return self.root / "objects"

    @property
    def sha256_objects_dir(self) -> Path:
        return self.objects_dir / "sha256"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / safe_name(run_id, "run_id")

    def artifacts_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "artifacts"

    def trajectories_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "trajectories"

    def configs_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "configs"

    def logs_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "logs"

    def knowledge_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "knowledge"

    def snapshot_manifests_dir(self, run_id: str) -> Path:
        return self.knowledge_dir(run_id) / "snapshots"

    def snapshot_pointers_dir(self, run_id: str) -> Path:
        return self.knowledge_dir(run_id) / "pointers"

    def snapshot_manifest_path(self, run_id: str, snapshot_id: str) -> Path:
        return self.snapshot_manifests_dir(run_id) / (
            safe_name(snapshot_id, "snapshot_id") + ".json"
        )

    def snapshot_pointer_path(self, run_id: str, name: str = "latest") -> Path:
        return self.snapshot_pointers_dir(run_id) / (
            safe_name(name, "pointer name") + ".json"
        )

    def checkpoints_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "checkpoints"

    def checkpoint_entries_dir(self, run_id: str) -> Path:
        return self.checkpoints_dir(run_id) / "entries"

    def checkpoint_pointers_dir(self, run_id: str) -> Path:
        return self.checkpoints_dir(run_id) / "pointers"

    def checkpoint_dir(self, run_id: str, checkpoint_id: str) -> Path:
        return self.checkpoint_entries_dir(run_id) / safe_name(
            checkpoint_id, "checkpoint_id"
        )

    def checkpoint_manifest_path(self, run_id: str, checkpoint_id: str) -> Path:
        return self.checkpoint_dir(run_id, checkpoint_id) / "manifest.json"

    def checkpoint_pointer_path(self, run_id: str, name: str = "latest") -> Path:
        return self.checkpoint_pointers_dir(run_id) / (
            safe_name(name, "pointer name") + ".json"
        )

    def artifact_path(
        self,
        run_id: str,
        artifact_id: str,
        *,
        suffix: str = "",
    ) -> Path:
        suffix = _validate_suffix(suffix)
        return self.artifacts_dir(run_id) / (
            safe_name(artifact_id, "artifact_id") + suffix
        )

    def object_path(self, digest: str, *, suffix: str = "") -> Path:
        """Return a sharded content-addressed path for a SHA-256 digest."""

        digest = _validate_sha256(digest)
        suffix = _validate_suffix(suffix)
        return self.sha256_objects_dir / digest[:2] / f"{digest}{suffix}"

    def ensure(self) -> StorageLayout:
        """Create the global layout roots and return ``self``."""

        for path in (self.root, self.sha256_objects_dir, self.runs_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def ensure_run(self, run_id: str) -> Path:
        """Create the stable directory skeleton for one run."""

        paths = (
            self.artifacts_dir(run_id),
            self.trajectories_dir(run_id),
            self.configs_dir(run_id),
            self.logs_dir(run_id),
            self.snapshot_manifests_dir(run_id),
            self.snapshot_pointers_dir(run_id),
            self.checkpoint_entries_dir(run_id),
            self.checkpoint_pointers_dir(run_id),
        )
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        return self.run_dir(run_id)

    def relative_uri(self, path: str | Path) -> str:
        """Encode a path below the root as a portable POSIX relative URI."""

        resolved = ensure_within(self.root, Path(path), "path")
        return resolved.relative_to(self.root).as_posix()

    def resolve_uri(self, uri: str, *, allow_external: bool = False) -> Path:
        """Resolve a relative or ``file:`` URI to a local path.

        By default a URI must remain below this layout's root. Network schemes are
        intentionally rejected because integrity checks operate on local bytes.
        """

        if not isinstance(uri, str) or not uri.strip():
            raise InvalidStoragePath("uri must be a non-empty string")
        parsed = urlparse(uri.strip())
        if parsed.scheme not in {"", "file"}:
            raise InvalidStoragePath(f"unsupported storage URI scheme: {parsed.scheme}")
        if parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise InvalidStoragePath("file URI cannot reference a remote host")
            candidate = Path(unquote(parsed.path))
        else:
            if parsed.netloc or parsed.query or parsed.fragment:
                raise InvalidStoragePath(
                    "relative storage URI cannot include URL parts"
                )
            relative = safe_relative_path(unquote(parsed.path), "uri")
            candidate = self.root / relative
        if allow_external:
            return candidate.expanduser().resolve(strict=False)
        return ensure_within(self.root, candidate, "uri")


__all__ = [
    "STORAGE_ROOT_ENV",
    "InvalidStoragePath",
    "StorageLayout",
    "ensure_within",
    "safe_name",
    "safe_relative_path",
]
