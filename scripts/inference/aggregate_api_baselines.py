"""Validate three independent complete baseline runs and report mean/sample std."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from spatialcraft.storage.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    sha256_file,
)

COUNTS = {"robospatial": 175, "erqa": 200, "omni3d": 250, "sat": 300}


def stats(values):
    return {
        "accuracy_runs": values,
        "n_runs": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "variance": statistics.variance(values),
        "std_ddof": 1,
        "std_population": statistics.pstdev(values),
        "mean_percent": statistics.mean(values) * 100,
        "std_percentage_points": statistics.stdev(values) * 100,
    }


def aggregate(runs, *, counts=None):
    counts = counts or COUNTS
    roots = [Path(p).resolve() for p in runs]
    if len(roots) != 3 or len(set(roots)) != 3:
        raise ValueError("Require three distinct run directories")
    for root in roots:
        summary = read_json(root / "results/summary.json")
        if summary.get("status") != "completed" or summary.get("failed_datasets"):
            raise ValueError(f"Incomplete run: {root}")
    datasets = {}
    source_checksums = {}
    requested_models = set()
    for name, total in counts.items():
        runs_rows = []
        previous_binding = None
        previous_ids = None
        previous_models = None
        all_response_ids = set()
        for root in roots:
            binding_path = root / name / "journal.json"
            binding = read_json(binding_path)["binding"]
            requested = binding.get("model_id")
            if not isinstance(requested, str) or not requested:
                raise ValueError("Missing model identity in experiment binding")
            requested_models.add(requested)
            if len(requested_models) != 1:
                raise ValueError("Cannot combine different requested models")
            report_path = root / name / "results/deployment.json"
            report = read_json(report_path)
            predictions = root / name / "results/predictions.jsonl"
            rows = [
                json.loads(line)
                for line in predictions.read_text().splitlines()
                if line
            ]
            ids = [row["task_id"] for row in rows]
            public_path = root / "data" / name / "public.jsonl"
            public_ids = [
                json.loads(line)["task_id"]
                for line in public_path.read_text().splitlines()
                if line
            ]
            manifest_path = root / "data/manifest.json"
            manifest = read_json(manifest_path)["datasets"][name]
            if (
                report.get("status") != "completed"
                or report.get("evaluated") != total
                or report.get("total") != total
                or len(rows) != total
                or len(set(ids)) != total
                or ids != public_ids
                or ids != manifest["task_ids"]
                or sha256_file(public_path) != manifest["public_sha256"]
            ):
                raise ValueError(f"Incomplete or misaligned predictions: {root}/{name}")
            if previous_binding is not None and binding != previous_binding:
                raise ValueError(f"Experiment protocol differs: {root}/{name}")
            if previous_ids is not None and ids != previous_ids:
                raise ValueError(f"Evaluation task IDs/order differ: {root}/{name}")
            models = sorted({row["model"] for row in rows})
            if any(
                model != requested
                and not re.fullmatch(
                    re.escape(requested) + r"-\d{4}-\d{2}-\d{2}", model
                )
                for model in models
            ):
                raise ValueError(
                    "Returned model is not the requested model or dated snapshot"
                )
            if previous_models is not None and models != previous_models:
                raise ValueError(f"Returned model snapshot differs: {root}/{name}")
            response_ids = [row["response_id"] for row in rows]
            if (
                any(not value for value in response_ids)
                or len(set(response_ids)) != total
                or all_response_ids.intersection(response_ids)
            ):
                raise ValueError(f"Responses reused across evaluations: {root}/{name}")
            if any(row["reward"] not in (0, 1) for row in rows):
                raise ValueError("Baseline rewards must be binary")
            correct = sum(row["reward"] for row in rows)
            if (
                correct != report["correct"]
                or not math.isclose(
                    correct / total, report["accuracy"], rel_tol=0, abs_tol=1e-12
                )
                or models != report["model_ids"]
            ):
                raise ValueError(
                    f"Accuracy/model report disagrees with predictions: {root}/{name}"
                )
            all_response_ids.update(response_ids)
            previous_binding, previous_ids, previous_models = binding, ids, models
            runs_rows.append(rows)
            for path in (
                binding_path,
                report_path,
                predictions,
                public_path,
                manifest_path,
            ):
                source_checksums[str(path)] = sha256_file(path)
        values = [sum(row["reward"] for row in rows) / total for rows in runs_rows]
        categories = {}
        for key in ("question_type", "answer_type", "source_answer_type"):
            groups = sorted({row[key] for row in runs_rows[0]})
            categories[key] = {}
            for group in groups:
                selected_ids = [
                    {row["task_id"] for row in rows if row[key] == group}
                    for rows in runs_rows
                ]
                if any(ids != selected_ids[0] for ids in selected_ids):
                    raise ValueError(
                        f"Category membership changed: {name}/{key}/{group}"
                    )
                size = len(selected_ids[0])
                category_values = [
                    sum(row["reward"] for row in rows if row[key] == group) / size
                    for rows in runs_rows
                ]
                categories[key][group] = {
                    "questions_per_run": size,
                    **stats(category_values),
                }
        datasets[name] = {
            "questions_per_run": total,
            "correct_runs": [
                int(sum(row["reward"] for row in rows)) for rows in runs_rows
            ],
            "model_ids": previous_models,
            **stats(values),
            "categories": categories,
        }
    return {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [str(root) for root in roots],
        "n_runs": 3,
        "model": next(iter(requested_models)),
        "run_labels": [root.name for root in roots],
        "api_seed": None,
        "seed_note": "Responses API has no generation seed. Run identifiers are repetition labels only; these are repeated evaluations at unchanged temperature=0, not controlled model-seed trials.",
        "units": {
            "mean": "accuracy fraction",
            "std": "accuracy fraction, sample standard deviation (ddof=1)",
            "variance": "squared accuracy fraction (ddof=1)",
            "std_percentage_points": "percentage points",
        },
        "datasets": datasets,
        "source_sha256": source_checksums,
    }


def write_results(output, result):
    output.mkdir(parents=True, exist_ok=True)
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(
        [
            "dataset",
            "questions_per_run",
            "run1_accuracy_percent",
            "run2_accuracy_percent",
            "run3_accuracy_percent",
            "mean_percent",
            "std_percentage_points_ddof1",
            "variance_accuracy_fraction_ddof1",
        ]
    )
    lines = [
        f"# {result['model']}: three repeated evaluations",
        "",
        "Responses API does not expose a generation seed; run numbers are repetition labels. Std is the sample standard deviation (ddof=1).",
        "",
        "| Dataset | Run 1 | Run 2 | Run 3 | Mean ± std |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in result["datasets"].items():
        values = [value * 100 for value in row["accuracy_runs"]]
        writer.writerow(
            [
                name,
                row["questions_per_run"],
                *values,
                row["mean_percent"],
                row["std_percentage_points"],
                row["variance"],
            ]
        )
        lines.append(
            f"| {name} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% | {row['mean_percent']:.2f} ± {row['std_percentage_points']:.2f}% |"
        )
    atomic_write_bytes(output / "accuracy_mean_std.csv", table.getvalue().encode())
    atomic_write_bytes(
        output / "accuracy_mean_std.md", ("\n".join(lines) + "\n").encode()
    )
    # JSON is written last as the completed aggregate's commit point.
    atomic_write_json(output / "accuracy_mean_std.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.runs)
    write_results(args.output, result)
    print(
        json.dumps(
            {
                name: {
                    key: row[key]
                    for key in ("accuracy_runs", "mean", "std", "variance")
                }
                for name, row in result["datasets"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
