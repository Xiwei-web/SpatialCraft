"""Crash-safe local I/O primitives for concurrent HPC workers."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

DEFAULT_CHUNK_SIZE = 1024 * 1024


class LockTimeoutError(TimeoutError):
    """Raised when an advisory file lock cannot be acquired in time."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically and reject non-standard numeric values."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (payload + "\n").encode("utf-8")


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    """Return the hexadecimal SHA-256 digest of an in-memory payload."""

    return hashlib.sha256(bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Stream a file into SHA-256 without loading it into memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync so a rename survives a host crash."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EROFS}:
            raise
    finally:
        os.close(descriptor)


def _commit_temp_file(temp_path: Path, target: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temp_path, target)
    else:
        # A hard-link commit provides atomic create-if-absent semantics. Both paths
        # are siblings, so they are guaranteed to reside on the same filesystem.
        os.link(temp_path, target)
        temp_path.unlink()
    _fsync_directory(target.parent)


def atomic_write_bytes(
    path: str | Path,
    data: bytes | bytearray | memoryview,
    *,
    overwrite: bool = True,
    mode: int = 0o644,
) -> Path:
    """Atomically replace or create a file with *data*.

    The temporary file is created beside the target so the final commit never
    crosses a filesystem boundary. File and parent-directory metadata are synced
    before returning.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(bytes(data))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, mode)
        _commit_temp_file(temp_path, target, overwrite=overwrite)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    overwrite: bool = True,
    mode: int = 0o644,
) -> Path:
    """Atomically write a text file."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return atomic_write_bytes(
        path,
        text.encode(encoding),
        overwrite=overwrite,
        mode=mode,
    )


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    overwrite: bool = True,
    mode: int = 0o644,
) -> Path:
    """Atomically write canonical UTF-8 JSON."""

    return atomic_write_bytes(
        path,
        canonical_json_bytes(value),
        overwrite=overwrite,
        mode=mode,
    )


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document."""

    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _copy_stream(source: BinaryIO, destination: BinaryIO, chunk_size: int) -> None:
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            return
        destination.write(chunk)


def atomic_copy(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    """Atomically copy one regular file while preserving its permission bits."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"source is not a regular file: {source_path}")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(temp_name)
    try:
        with (
            source_path.open("rb") as source_stream,
            os.fdopen(descriptor, "wb") as destination_stream,
        ):
            _copy_stream(source_stream, destination_stream, chunk_size)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
        shutil.copymode(source_path, temp_path, follow_symlinks=True)
        _commit_temp_file(temp_path, target, overwrite=overwrite)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return target


@contextmanager
def file_lock(
    path: str | Path,
    *,
    timeout: float | None = 60.0,
    poll_interval: float = 0.1,
) -> Iterator[Path]:
    """Acquire an exclusive advisory lock backed by ``fcntl.flock``.

    Lock files are intentionally persistent; the kernel releases the actual lock
    when a worker exits, including after a crash.
    """

    if timeout is not None and timeout < 0:
        raise ValueError("timeout cannot be negative")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    started = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise LockTimeoutError(
                        f"timed out waiting for lock: {lock_path}"
                    ) from exc
                time.sleep(poll_interval)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def staged_directory(target: str | Path) -> Iterator[Path]:
    """Build an immutable directory privately, then rename it into place.

    The target must not already exist. Callers coordinating multiple writers
    should hold :func:`file_lock` for the check and commit interval.
    """

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target_path.name}.",
            suffix=".staging",
            dir=target_path.parent,
        )
    )
    try:
        yield staging
        if target_path.exists():
            raise FileExistsError(f"target directory already exists: {target_path}")
        os.rename(staging, target_path)
        _fsync_directory(target_path.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "LockTimeoutError",
    "atomic_copy",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "canonical_json_bytes",
    "file_lock",
    "read_json",
    "sha256_bytes",
    "sha256_file",
    "staged_directory",
]
