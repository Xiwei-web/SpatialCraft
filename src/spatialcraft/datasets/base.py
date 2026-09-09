"""Dataset adapter contracts and streaming record helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar

from spatialcraft.schemas import TaskSample, TaskSplit
from spatialcraft.schemas._base import require_non_empty


class DatasetError(RuntimeError):
    """Base class for dataset-layer failures."""


class DatasetConfigurationError(DatasetError):
    """Raised when a dataset root or adapter configuration is invalid."""


class DatasetFormatError(DatasetError):
    """Raised when source data cannot be normalized safely."""


class UnknownSplitError(DatasetError):
    """Raised when an adapter does not expose a requested split."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetInfo:
    """Static metadata for one configured dataset adapter."""

    name: str
    root: Path
    available_splits: tuple[str, ...]
    default_split: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_non_empty(self.name, "name"))
        object.__setattr__(self, "root", Path(self.root).expanduser())
        object.__setattr__(self, "available_splits", tuple(self.available_splits))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.available_splits:
            raise ValueError("available_splits cannot be empty")
        if self.default_split not in self.available_splits:
            raise ValueError("default_split must be listed in available_splits")
        if len(self.available_splits) != len(set(self.available_splits)):
            raise ValueError("available_splits cannot contain duplicates")


_SPLIT_ALIASES = {
    "dev": "validation",
    "eval": "test",
    "evaluation": "test",
    "testing": "test",
    "training": "train",
    "val": "validation",
}


class DatasetAdapter(ABC):
    """Convert one source dataset into streaming :class:`TaskSample` objects."""

    dataset_name: ClassVar[str]
    available_splits: ClassVar[tuple[str, ...]]
    default_split: ClassVar[str]

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.dataset_name,
            root=self.root,
            available_splits=self.available_splits,
            default_split=self.default_split,
        )

    def normalize_split(self, split: str | TaskSplit | None) -> str:
        if split is None:
            return self.default_split
        value = split.value if isinstance(split, TaskSplit) else str(split)
        value = _SPLIT_ALIASES.get(value.strip().lower(), value.strip().lower())
        if value not in self.available_splits:
            choices = ", ".join(self.available_splits)
            raise UnknownSplitError(
                f"{self.dataset_name} does not expose split {value!r}; "
                f"available: {choices}"
            )
        return value

    def canonical_task_split(self, split: str) -> TaskSplit:
        if split == "train" or split == "static":
            return TaskSplit.TRAIN
        if split == "validation":
            return TaskSplit.VALIDATION
        if split in {"test", "context", "configuration", "compatibility"}:
            return TaskSplit.TEST
        return TaskSplit.UNSPECIFIED

    @abstractmethod
    def iter_samples(
        self,
        split: str | TaskSplit | None = None,
        *,
        limit: int | None = None,
    ) -> Iterator[TaskSample]:
        """Yield normalized samples without loading the full split into memory."""

    def load_split(
        self,
        split: str | TaskSplit | None = None,
        *,
        limit: int | None = None,
    ) -> tuple[TaskSample, ...]:
        return tuple(self.iter_samples(split, limit=limit))

    def get_sample(
        self, index: int, split: str | TaskSplit | None = None
    ) -> TaskSample:
        if index < 0:
            raise IndexError("dataset index cannot be negative")
        sample = next(islice(self.iter_samples(split), index, index + 1), None)
        if sample is None:
            raise IndexError(
                f"sample index {index} is out of range for {self.dataset_name}"
            )
        return sample

    def _validate_limit(self, limit: int | None) -> None:
        if limit is not None and limit < 0:
            raise ValueError("limit cannot be negative")

    def _validate_sample(self, sample: TaskSample, split: str) -> TaskSample:
        if sample.dataset != self.dataset_name:
            raise DatasetFormatError(
                f"adapter {self.dataset_name} produced dataset={sample.dataset!r}"
            )
        expected = self.canonical_task_split(split)
        if sample.split is not expected:
            raise DatasetFormatError(
                f"adapter produced split={sample.split.value!r}; "
                f"expected {expected.value!r}"
            )
        return sample


class RecordDatasetAdapter(DatasetAdapter):
    """Adapter template for row-oriented JSON or Parquet datasets."""

    @abstractmethod
    def iter_records(self, split: str) -> Iterator[Mapping[str, Any]]:
        """Yield raw records for a normalized source split name."""

    @abstractmethod
    def normalize_record(
        self, record: Mapping[str, Any], *, index: int, split: str
    ) -> TaskSample:
        """Normalize one source record."""

    def iter_samples(
        self,
        split: str | TaskSplit | None = None,
        *,
        limit: int | None = None,
    ) -> Iterator[TaskSample]:
        self._validate_limit(limit)
        normalized_split = self.normalize_split(split)
        if limit == 0:
            return
        for index, record in enumerate(self.iter_records(normalized_split)):
            if limit is not None and index >= limit:
                break
            yield self._validate_sample(
                self.normalize_record(record, index=index, split=normalized_split),
                normalized_split,
            )


def iter_parquet_records(
    path: str | Path, *, batch_size: int = 32
) -> Iterator[Mapping[str, Any]]:
    """Stream Parquet rows and import PyArrow only when an adapter is used."""

    source = Path(path)
    if not source.is_file():
        raise DatasetConfigurationError(f"Parquet source not found: {source}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetConfigurationError(
            "PyArrow is required to read benchmark Parquet files"
        ) from exc
    try:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetFormatError(f"cannot read Parquet source {source}: {exc}") from exc


__all__ = [
    "DatasetAdapter",
    "DatasetConfigurationError",
    "DatasetError",
    "DatasetFormatError",
    "DatasetInfo",
    "RecordDatasetAdapter",
    "UnknownSplitError",
    "iter_parquet_records",
]
