"""Deterministic dataset partitioning and distribution-matched sampling."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from typing import TypeVar

from spatialcraft.schemas import TaskSample, TaskSplit

ItemT = TypeVar("ItemT")
KeyFn = Callable[[ItemT], str]


def _stable_seed(seed: int, namespace: str) -> int:
    payload = f"{seed}\0{namespace}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


class SplitManager:
    """Reproducible sampling that is independent of global RNG state."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def _rng(self, namespace: str) -> random.Random:
        return random.Random(_stable_seed(self.seed, namespace))

    def sample(
        self, items: Sequence[ItemT], count: int, *, namespace: str = "sample"
    ) -> tuple[ItemT, ...]:
        if count < 0:
            raise ValueError("sample count cannot be negative")
        if count > len(items):
            raise ValueError(f"cannot sample {count} items from {len(items)}")
        indices = list(range(len(items)))
        self._rng(namespace).shuffle(indices)
        return tuple(items[index] for index in indices[:count])

    def sample_matched(
        self,
        candidates: Sequence[ItemT],
        counts: Mapping[str, int],
        *,
        key: KeyFn[ItemT],
        namespace: str = "matched",
    ) -> tuple[ItemT, ...]:
        """Sample exact per-category counts or fail without partial output."""

        buckets: dict[str, list[ItemT]] = defaultdict(list)
        for item in candidates:
            buckets[str(key(item))].append(item)
        selected: list[ItemT] = []
        for category in sorted(counts):
            requested = int(counts[category])
            if requested < 0:
                raise ValueError("matched sample counts cannot be negative")
            bucket = buckets.get(category, [])
            if len(bucket) < requested:
                raise ValueError(
                    f"category {category!r} has {len(bucket)} candidates; "
                    f"requires {requested}"
                )
            selected.extend(
                self.sample(
                    bucket,
                    requested,
                    namespace=f"{namespace}:{category}",
                )
            )
        self._rng(f"{namespace}:merge").shuffle(selected)
        return tuple(selected)

    def sample_matching_reference(
        self,
        candidates: Sequence[ItemT],
        reference: Iterable[ItemT],
        *,
        key: KeyFn[ItemT],
        namespace: str = "matching-reference",
    ) -> tuple[ItemT, ...]:
        counts = Counter(str(key(item)) for item in reference)
        return self.sample_matched(candidates, counts, key=key, namespace=namespace)

    def partition_tasks(
        self,
        samples: Sequence[TaskSample],
        *,
        train: float,
        validation: float,
        test: float,
        group_key: Callable[[TaskSample], str] | None = None,
        namespace: str = "partition",
    ) -> dict[TaskSplit, tuple[TaskSample, ...]]:
        """Partition tasks while keeping grouped variants in the same split."""

        ratios = (float(train), float(validation), float(test))
        if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("partition ratios must be non-negative and sum to 1")
        key_fn = group_key or (lambda sample: sample.source_id or sample.task_id)
        groups: dict[str, list[TaskSample]] = defaultdict(list)
        for sample in samples:
            groups[str(key_fn(sample))].append(sample)
        keys = sorted(groups)
        self._rng(namespace).shuffle(keys)
        train_end = round(len(keys) * ratios[0])
        validation_end = train_end + round(len(keys) * ratios[1])
        assignments = {
            TaskSplit.TRAIN: keys[:train_end],
            TaskSplit.VALIDATION: keys[train_end:validation_end],
            TaskSplit.TEST: keys[validation_end:],
        }
        return {
            split: tuple(
                replace(sample, split=split)
                for group in selected_keys
                for sample in groups[group]
            )
            for split, selected_keys in assignments.items()
        }


def task_category(sample: TaskSample) -> str:
    value = sample.metadata.get("question_type") or sample.metadata.get("category")
    if value is None:
        raise ValueError(f"task {sample.task_id} has no category metadata")
    return str(value)


__all__ = ["SplitManager", "task_category"]
