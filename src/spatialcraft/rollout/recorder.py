"""Concurrent-safe JSONL trajectory recording and strict replay."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from spatialcraft.schemas import Trajectory
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import file_lock


class TrajectoryRecordError(RuntimeError):
    pass


class TrajectoryRecorder:
    def __init__(
        self,
        layout: StorageLayout,
        run_id: str,
        *,
        filename: str = "trajectories.jsonl",
    ) -> None:
        self.layout = layout
        self.run_id = run_id
        self.layout.ensure_run(run_id)
        if Path(filename).name != filename or not filename.endswith(".jsonl"):
            raise ValueError("trajectory filename must be a .jsonl basename")
        self.path = self.layout.trajectories_dir(run_id) / filename
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def append(self, trajectory: Trajectory) -> Path:
        line = trajectory.to_json() + "\n"
        with (
            file_lock(self.lock_path),
            self.path.open("a", encoding="utf-8") as stream,
        ):
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return self.path

    def replay(self) -> Iterator[Trajectory]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError("trajectory line is not a JSON object")
                    yield Trajectory.from_dict(value)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise TrajectoryRecordError(
                        f"invalid trajectory JSONL at line {line_number}: {exc}"
                    ) from exc


__all__ = ["TrajectoryRecordError", "TrajectoryRecorder"]
