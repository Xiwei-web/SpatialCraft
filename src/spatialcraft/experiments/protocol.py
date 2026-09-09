"""Explicit instance-level SMA/draft split construction for three benchmarks."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from spatialcraft.schemas import TaskSample, TaskSplit


def split_category(task: TaskSample) -> str:
    # SMA Appendix C.2: Omni3D has answer-type categories, not spatial taxonomy.
    key = "source_answer_type" if task.dataset == "omni3d" else "question_type"
    value = task.metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {key} for {task.dataset}/{task.task_id}")
    return value


def stratified_halves(
    tasks: tuple[TaskSample, ...], *, seed: int = 42
) -> tuple[tuple[TaskSample, ...], tuple[TaskSample, ...]]:
    """50/50 within category, alternating odd remainder starting environment.

    A single Python Random(seed) shuffles each category in sorted category order.
    Members start in sorted task-id order. This fixes details not specified by
    the papers; it is not a claim to reproduce SMA's exact unpublished split IDs.
    Different questions on the same image may cross splits, as this protocol is
    instance-level. Preparation separately reports shared-image overlap.
    """
    if not tasks or len({t.dataset for t in tasks}) != 1:
        raise ValueError("Expected one nonempty dataset")
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    buckets: dict[str, list[TaskSample]] = defaultdict(list)
    for task in tasks:
        buckets[split_category(task)].append(task)
    rng = random.Random(seed)
    environment: list[TaskSample] = []
    deployment: list[TaskSample] = []
    extra_to_environment = True
    for category in sorted(buckets):
        rows = sorted(buckets[category], key=lambda t: t.task_id)
        rng.shuffle(rows)
        midpoint = len(rows) // 2
        if len(rows) % 2:
            midpoint += int(extra_to_environment)
            extra_to_environment = not extra_to_environment
        for name, split, selected, output in (
            ("environment", TaskSplit.TRAIN, rows[:midpoint], environment),
            ("deployment", TaskSplit.TEST, rows[midpoint:], deployment),
        ):
            output.extend(
                replace(
                    row,
                    split=split,
                    metadata={**row.metadata, "experiment_split": name},
                )
                for row in selected
            )
    assert len(environment) + len(deployment) == len(tasks)
    assert not {r.task_id for r in environment} & {r.task_id for r in deployment}
    return tuple(environment), tuple(deployment)


def public_task(task: TaskSample) -> dict:
    """Executor input without GT, target masks, reference depth, or raw metadata."""
    value = task.to_dict()
    value["reference_answer"] = None
    # Image provenance from adapters can also contain annotations; the executor
    # needs pixels, dimensions and media identifiers, not arbitrary metadata.
    for image in value["images"]:
        image["metadata"] = {}
    value["metadata"] = {
        key: task.metadata[key]
        for key in (
            "question_type",
            "source_answer_type",
            "choice_labels",
            "visual_indices",
            "experiment_split",
        )
        if key in task.metadata
    }
    return value
