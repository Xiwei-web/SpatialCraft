"""Persistent seed42 per-category environment/deployment split for ViewSpatial."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from spatialcraft.datasets import create_default_registry
from spatialcraft.schemas import TaskSample
from spatialcraft.storage.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    file_lock,
    read_json,
    sha256_file,
)

from .journal import digest
from .protocol import public_task, split_category, stratified_halves

SPLIT_PROTOCOL = {
    "seed": 42,
    "category_key": "question_type",
    "environment_fraction": 0.5,
    "unit": "task_instance_not_unique_image",
    "algorithm": "sorted categories; sorted task IDs; one Python random.Random(42); shuffle each category; odd remainders alternate environment first",
}


def rows(path):
    return tuple(
        TaskSample.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line
    )


def write_split(root, tasks, sources):
    if len(tasks) % 2 or any(t.dataset != "viewspatial" for t in tasks):
        raise ValueError("Expected an even-sized ViewSpatial dataset")
    environment, deployment = stratified_halves(tasks, seed=42)
    records = {}
    for name, selected in (("environment", environment), ("deployment", deployment)):
        for folder, values in (
            ("splits", [public_task(t) for t in selected]),
            ("verification", [t.to_dict() for t in selected]),
        ):
            relative = f"viewspatial/{folder}/{name}.jsonl"
            path = root / relative
            content = b"".join(canonical_json_bytes(value) for value in values)
            if path.exists():
                if path.read_bytes() != content:
                    raise ValueError(
                        "Existing split differs; use a new preparation directory"
                    )
            else:
                atomic_write_bytes(path, content, mode=0o600, overwrite=False)
            records[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    image_sets = [
        {image.sha256 or image.uri for task in selected for image in task.images}
        for selected in (environment, deployment)
    ]
    manifest = {
        "schema_version": 1,
        "dataset": "viewspatial",
        "split_protocol": SPLIT_PROTOCOL,
        "sources": sources,
        "total": len(tasks),
        "environment": len(environment),
        "deployment": len(deployment),
        "categories": {
            name: dict(sorted(Counter(map(split_category, selected)).items()))
            for name, selected in (
                ("all", tasks),
                ("environment", environment),
                ("deployment", deployment),
            )
        },
        "task_ids": {
            "environment": [t.task_id for t in environment],
            "deployment": [t.task_id for t in deployment],
        },
        "shared_image_count": len(image_sets[0] & image_sets[1]),
        "input_files": records,
    }
    path = root / "manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Existing split manifest changed")
    else:
        atomic_write_json(path, manifest, overwrite=False, mode=0o600)
    return manifest


def validate_split(root, sources, expected_total=5712):
    manifest = read_json(root / "manifest.json")
    if (
        manifest["split_protocol"] != SPLIT_PROTOCOL
        or manifest["sources"] != sources
        or manifest["total"] != expected_total
        or manifest["deployment"] != expected_total // 2
        or manifest["environment"] != expected_total // 2
    ):
        raise ValueError("ViewSpatial preparation source/protocol/count mismatch")
    for name, record in manifest["input_files"].items():
        if sha256_file(root / name) != record["sha256"]:
            raise ValueError("ViewSpatial prepared input checksum mismatch")
    private = {}
    for split in ("environment", "deployment"):
        private[split] = rows(root / f"viewspatial/verification/{split}.jsonl")
        public = rows(root / f"viewspatial/splits/{split}.jsonl")
        if len(public) != expected_total // 2 or len(private[split]) != len(public):
            raise ValueError("Wrong number of ViewSpatial split rows")
        if [t.task_id for t in public] != manifest["task_ids"][split]:
            raise ValueError("ViewSpatial split task IDs changed")
        if any(
            digest(left.to_dict()) != digest(public_task(right))
            for left, right in zip(public, private[split], strict=True)
        ):
            raise ValueError("ViewSpatial public/private task mismatch")
    combined = (*private["environment"], *private["deployment"])
    expected = stratified_halves(combined, seed=42)
    for split, tasks in zip(("environment", "deployment"), expected, strict=True):
        if [t.task_id for t in tasks] != manifest["task_ids"][split]:
            raise ValueError(
                "ViewSpatial split does not implement seed42 category halves"
            )
    return manifest


def prepare_viewspatial(root: Path, benchmark_root: Path):
    source = benchmark_root / "ViewSpatial"
    sources = {
        name: sha256_file(source / name)
        for name in ("ViewSpatial-Bench.json", "scannetv2_val.zip", "val2017.zip")
    }
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with file_lock(root / ".preparation.lock", timeout=0):
        if not (root / "manifest.json").exists():
            adapter = create_default_registry(benchmark_root).create("viewspatial")
            tasks = []
            try:
                for task in adapter.iter_samples("test"):
                    tasks.append(task)
                    if len(tasks) % 500 == 0:
                        print(
                            f"ViewSpatial data preparation: {len(tasks)}/5712 normalized records",
                            flush=True,
                        )
            finally:
                adapter.close()
            if len(tasks) != 5712:
                raise ValueError(
                    f"Expected 5712 ViewSpatial source records, found {len(tasks)}"
                )
            write_split(root, tuple(tasks), sources)
        return validate_split(root, sources)
