"""Prepare the five-benchmark protocol; SAT uses validation-to-test transfer."""

import argparse
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

from spatialcraft.datasets import create_default_registry
from spatialcraft.datasets.normalizer import ImageMaterializer
from spatialcraft.schemas import TaskSplit
from spatialcraft.storage.atomic_io import canonical_json_bytes, file_lock, sha256_bytes

from .prepare import _content_keys, _image_keys, _immutable, file_record
from .protocol import public_task, split_category, stratified_halves

DATASETS = ("robospatial", "erqa", "omni3d", "sat", "viewspatial")


def sat_splits(validation, test, seed=42):
    if not test or len(validation) < len(test):
        raise ValueError(
            "SAT needs a validation pool at least as large as the official test set"
        )
    if any(row.dataset != "sat" for row in (*validation, *test)):
        raise ValueError("SAT split sources have inconsistent dataset identity")
    if len({r.task_id for r in validation}) != len(validation) or len(
        {r.task_id for r in test}
    ) != len(test):
        raise ValueError("Duplicate SAT source task IDs")
    if {r.task_id for r in validation} & {r.task_id for r in test}:
        raise ValueError("SAT validation/test identities overlap")
    selected = random.Random(seed).sample(
        sorted(validation, key=lambda r: r.task_id), len(test)
    )
    # Test labels are never inspected when sampling environment tasks.
    return tuple(
        replace(
            r,
            split=TaskSplit.TRAIN,
            metadata={**r.metadata, "experiment_split": "environment"},
        )
        for r in selected
    ), tuple(
        replace(
            r,
            split=TaskSplit.TEST,
            metadata={**r.metadata, "experiment_split": "deployment"},
        )
        for r in test
    )


def write_preparation(output, sources, *, sat_validation=(), seed=42, provenance=None):
    names = tuple(name for name in DATASETS if name in sources)
    if not names or set(sources) - set(DATASETS):
        raise ValueError("Select supported datasets")
    payloads, counts = {}, {}
    for name in names:
        rows = tuple(sources[name])
        if any(r.dataset != name for r in rows):
            raise ValueError("Dataset source identity mismatch")
        environment, deployment = (
            sat_splits(tuple(sat_validation), rows, seed)
            if name == "sat"
            else stratified_halves(rows, seed=seed)
        )
        for split, values in (("environment", environment), ("deployment", deployment)):
            for folder, data in (
                ("splits", [public_task(r) for r in values]),
                ("verification", [r.to_dict() for r in values]),
            ):
                payloads[f"{name}/{folder}/{split}.jsonl"] = b"".join(
                    canonical_json_bytes(r) for r in data
                )
        counts[name] = {
            "total": len(environment) + len(deployment),
            "environment": len(environment),
            "deployment": len(deployment),
            "categories": {
                split: dict(Counter(map(split_category, values)))
                for split, values in (
                    ("environment", environment),
                    ("deployment", deployment),
                )
            },
            "shared_image_count": len(
                _image_keys(environment) & _image_keys(deployment)
            ),
            "shared_question_image_choices_count": len(
                _content_keys(environment) & _content_keys(deployment)
            ),
            "maximum_images_per_task": max(
                len(r.images) for r in (*environment, *deployment)
            ),
        }
    manifest = {
        "schema_version": 2,
        "status": "prepared_not_run",
        "dataset_order": list(names),
        "datasets": counts,
        "split_protocol": {
            "seed": seed,
            "non_sat": "stratified_halves_by_instance_v1",
            "sat": "sorted_validation_task_ids_random_sample_test_size_then_official_test",
            "sat_validation_pool_count": len(sat_validation),
            "test_labels_used_for_sampling": False,
            "exact_unpublished_SMA_split_claimed": False,
        },
        "provenance": dict(provenance or {}),
        "input_files": {
            name: {"sha256": sha256_bytes(value), "size_bytes": len(value)}
            for name, value in sorted(payloads.items())
        },
    }
    output = Path(output)
    with file_lock(output / ".preparation.lock"):
        if (output / "manifest.json").exists() and (
            output / "manifest.json"
        ).read_bytes() != canonical_json_bytes(manifest):
            raise ValueError("Preparation binding changed; use a new output directory")
        for name, value in payloads.items():
            _immutable(output / name, value, private="/verification/" in name)
        _immutable(output / "manifest.json", canonical_json_bytes(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("/l/users/xiwei.liu/benchmark")
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate dataset names")
    registry = create_default_registry(args.benchmark_root)
    sources, provenance, validation = {}, {}, ()
    for name in args.datasets:
        adapter = registry.create(name)
        # New media goes into the selected preparation directory, leaving the source benchmark intact.
        if hasattr(adapter, "images"):
            adapter.images = ImageMaterializer(args.output / "media")
        print("PREPARE", name, flush=True)
        try:
            sources[name] = adapter.load_split("test")
            if name == "sat":
                validation = adapter.load_split("validation")
                selected = (
                    adapter.root / "SAT_val.parquet",
                    adapter.root / "SAT_test_circular_300.parquet",
                )
                if not selected[1].is_file():
                    raise ValueError(
                        "SAT formal preparation requires the declared circular 300 test source"
                    )
            elif name == "viewspatial":
                selected = tuple(
                    sorted(
                        p
                        for p in adapter.root.iterdir()
                        if p.suffix in {".json", ".zip"}
                    )
                )
                if not selected:
                    raise ValueError(
                        "Missing ViewSpatial source JSON/archive provenance"
                    )
            else:
                selected = tuple(sorted(adapter.root.rglob("*.parquet")))
            provenance[name] = [file_record(path) for path in selected]
        finally:
            if hasattr(adapter, "close"):
                adapter.close()
    report = write_preparation(
        args.output,
        sources,
        sat_validation=validation,
        seed=args.seed,
        provenance=provenance,
    )
    print(
        {
            name: {
                key: value
                for key, value in item.items()
                if key in ("environment", "deployment", "shared_image_count")
            }
            for name, item in report["datasets"].items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
