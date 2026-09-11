"""Prepare the five-benchmark protocol; SAT uses validation-to-test transfer."""

import argparse
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

from spatialcraft.datasets import create_default_registry
from spatialcraft.datasets.normalizer import ImageMaterializer
from spatialcraft.schemas import TaskSplit
from spatialcraft.storage.atomic_io import (
    canonical_json_bytes,
    file_lock,
    sha256_bytes,
    sha256_file,
)

from .prepare import _content_keys, _image_keys, _immutable, file_record
from .protocol import public_task, split_category, stratified_halves

DATASETS = ("robospatial", "erqa", "omni3d", "sat", "viewspatial")


EXACT_DUPLICATE_POLICY = "reject_cross_split_v1"
CONTENT_FINGERPRINT = "question_ordered_image_sha256_choices_v1"


def public_content_fingerprints(tasks, *, image_hashes=None):
    """Content-only fingerprints from actual local image bytes, never labels/IDs.

    Image ordering and exact question/choice text matter. Shared images with
    different questions are not duplicate tasks. Strict mode requires readable
    local media; it does not download or trust an unverified image URI/hash.
    """
    hashes = {} if image_hashes is None else image_hashes
    fingerprints = {}
    for task in tasks:
        images = []
        for image in task.images:
            if image.uri not in hashes:
                path = Path(image.uri)
                if not path.is_file():
                    raise ValueError(
                        "Strict content isolation requires readable local task images"
                    )
                hashes[image.uri] = sha256_file(path)
            checksum = hashes[image.uri]
            if image.sha256 is not None and checksum != image.sha256:
                raise ValueError("Strict content isolation image checksum changed")
            images.append(checksum)
        fingerprints[task.task_id] = sha256_bytes(
            canonical_json_bytes(
                {
                    "question": task.question,
                    "images": images,
                    "choices": task.choices,
                }
            )
        )
    return fingerprints


def strict_content_isolation(environment, deployment):
    hashes = {}
    training = public_content_fingerprints(environment, image_hashes=hashes)
    heldout = public_content_fingerprints(deployment, image_hashes=hashes)
    duplicates = set(training.values()) & set(heldout.values())
    if duplicates:
        raise ValueError(
            f"Exact public task content overlaps training/deployment ({len(duplicates)} groups); strict policy rejects this split without repartitioning"
        )
    shared_images = len(
        {hashes[image.uri] for task in environment for image in task.images}
        & {hashes[image.uri] for task in deployment for image in task.images}
    )
    return {
        "policy": EXACT_DUPLICATE_POLICY,
        "fingerprint": CONTENT_FINGERPRINT,
        "training_task_content_sha256": training,
        "deployment_task_content_sha256": heldout,
        "shared_image_count": shared_images,
        "shared_question_image_choices_count": 0,
    }


def validate_content_isolation(manifest, name, environment, deployment):
    """Recompute an explicit strict claim; absence keeps historical report-only policy."""
    protocol = manifest.get("split_protocol", {})
    policy = protocol.get("exact_duplicate_policy", "report_only")
    if policy == "report_only":
        return None
    if (
        policy != EXACT_DUPLICATE_POLICY
        or protocol.get("content_fingerprint") != CONTENT_FINGERPRINT
    ):
        raise ValueError("Unsupported exact-duplicate isolation declaration")
    audit = strict_content_isolation(environment, deployment)
    for key in ("shared_image_count", "shared_question_image_choices_count"):
        if manifest["datasets"][name].get(key) != audit[key]:
            raise ValueError(
                "Declared content isolation counts differ from actual prepared inputs"
            )
    return audit


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


def write_preparation(
    output,
    sources,
    *,
    sat_validation=(),
    seed=42,
    provenance=None,
    exact_duplicate_policy="report_only",
):
    if exact_duplicate_policy not in {"report_only", EXACT_DUPLICATE_POLICY}:
        raise ValueError("Unknown exact-duplicate preparation policy")
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
        isolation = (
            strict_content_isolation(environment, deployment)
            if exact_duplicate_policy == EXACT_DUPLICATE_POLICY
            else None
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
        if isolation is not None:
            for key in ("shared_image_count", "shared_question_image_choices_count"):
                counts[name][key] = isolation[key]
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
    if exact_duplicate_policy == EXACT_DUPLICATE_POLICY:
        manifest["split_protocol"].update(
            exact_duplicate_policy=EXACT_DUPLICATE_POLICY,
            content_fingerprint=CONTENT_FINGERPRINT,
            repartitioned_for_duplicates=False,
        )
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
    parser.add_argument(
        "--exact-duplicate-policy",
        choices=("report_only", EXACT_DUPLICATE_POLICY),
        default="report_only",
        help="Opt-in strict rejection of cross-split exact public task duplicates; never silently repartitions",
    )
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
        exact_duplicate_policy=args.exact_duplicate_policy,
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
