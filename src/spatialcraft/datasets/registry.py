"""Named dataset-adapter registry with lazy default construction."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .base import DatasetAdapter, DatasetConfigurationError

BENCHMARK_ROOT_ENV = "SPATIALCRAFT_BENCHMARK_ROOT"
DEFAULT_BENCHMARK_ROOT = Path("/l/users/xiwei.liu/benchmark")
AdapterFactory = Callable[[Path], DatasetAdapter]


@dataclass(frozen=True, slots=True)
class _Registration:
    canonical_name: str
    subdirectory: str
    factory: AdapterFactory


class DatasetRegistry:
    """Resolve stable dataset names and aliases to configured adapters."""

    def __init__(self, benchmark_root: str | Path | None = None) -> None:
        selected = benchmark_root or os.environ.get(BENCHMARK_ROOT_ENV)
        self.benchmark_root = Path(selected or DEFAULT_BENCHMARK_ROOT).expanduser()
        self._entries: dict[str, _Registration] = {}

    @staticmethod
    def _key(name: str) -> str:
        value = name.strip().lower().replace("_", "-")
        if not value:
            raise ValueError("dataset name cannot be empty")
        return value

    def register(
        self,
        name: str,
        factory: AdapterFactory,
        *,
        subdirectory: str,
        aliases: Iterable[str] = (),
        replace: bool = False,
    ) -> None:
        canonical = self._key(name)
        registration = _Registration(canonical, subdirectory, factory)
        keys = {canonical, *(self._key(alias) for alias in aliases)}
        collisions = sorted(key for key in keys if key in self._entries)
        if collisions and not replace:
            raise DatasetConfigurationError(
                f"dataset aliases already registered: {collisions}"
            )
        for key in keys:
            self._entries[key] = registration

    def names(self) -> tuple[str, ...]:
        return tuple(sorted({entry.canonical_name for entry in self._entries.values()}))

    def create(self, name: str, *, root: str | Path | None = None) -> DatasetAdapter:
        key = self._key(name)
        try:
            entry = self._entries[key]
        except KeyError as exc:
            raise DatasetConfigurationError(f"unknown dataset adapter: {name}") from exc
        dataset_root = (
            Path(root).expanduser()
            if root is not None
            else self.benchmark_root / entry.subdirectory
        )
        return entry.factory(dataset_root)


def create_default_registry(
    benchmark_root: str | Path | None = None,
) -> DatasetRegistry:
    """Register all built-in adapters without importing them at package startup."""

    from .adapters import (
        ERQAAdapter,
        Omni3DAdapter,
        RoboSpatialAdapter,
        SATAdapter,
        ViewSpatialAdapter,
    )

    registry = DatasetRegistry(benchmark_root)
    registry.register(
        "robospatial",
        RoboSpatialAdapter,
        subdirectory="RoboSpatial",
        aliases=("robospatial-home", "robo-spatial"),
    )
    registry.register("erqa", ERQAAdapter, subdirectory="ERQA")
    registry.register(
        "omni3d",
        Omni3DAdapter,
        subdirectory="Omni3D",
        aliases=("omni3d-bench",),
    )
    registry.register("sat", SATAdapter, subdirectory="SAT")
    registry.register(
        "viewspatial",
        ViewSpatialAdapter,
        subdirectory="ViewSpatial",
        aliases=("viewspatial-bench", "view-spatial"),
    )
    return registry


__all__ = [
    "BENCHMARK_ROOT_ENV",
    "DEFAULT_BENCHMARK_ROOT",
    "AdapterFactory",
    "DatasetRegistry",
    "create_default_registry",
]
