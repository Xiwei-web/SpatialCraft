"""Bounded real-tool artifact-chain smoke on a prepared environment image.

No LLM, API, labels, training, or deployment evaluation. Each tool invocation runs
in a fresh process to release GPU allocations and isolate third-party imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_INPUT = Path(
    "/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1/robospatial/splits/environment.jsonl"
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New diagnostic directory, not an old result directory",
    )
    result.add_argument("--environment-jsonl", type=Path, default=DEFAULT_INPUT)
    result.add_argument("--queries", nargs="+", default=["toilet", "frame"])
    result.add_argument("--tool-timeout", type=int, default=240)
    result.add_argument("--total-timeout", type=int, default=1500)
    result.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return result


def worker(args):
    from spatialcraft.schemas import ToolCall
    from spatialcraft.storage import StorageLayout
    from spatialcraft.tools import ArtifactStore, ToolExecutor
    from spatialcraft.tools.real import create_real_tool_registry

    request = json.loads(args.worker_request.read_text())
    if request["tool_name"] in {"detect", "segment", "reconstruct", "scale", "pose"}:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Real model smoke requires the allocated CUDA device")
        torch.set_num_threads(1)
    executor = ToolExecutor(
        create_real_tool_registry(),
        ArtifactStore(StorageLayout(args.output / "storage"), "tools_v2"),
    )
    result = executor.execute(ToolCall(**request))
    args.worker_result.write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False)
    )


def main():
    args = parser().parse_args()
    if args.worker_request:
        worker(args)
        return
    if args.tool_timeout <= 0 or not 0 < args.total_timeout <= 1800:
        raise SystemExit("Use positive tool timeout and total timeout <=1800 seconds")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    if report_path.exists():
        raise SystemExit(
            "Diagnostic report already exists; choose a new output directory"
        )
    with args.environment_jsonl.open() as stream:
        task = json.loads(next(stream))
    image = Path(task["images"][0]["uri"]).resolve(strict=True)
    started = time.monotonic()
    project = Path(__file__).resolve().parents[2]
    source_files = [
        Path(__file__).resolve(),
        *sorted((project / "src/spatialcraft/tools").rglob("*.py")),
    ]
    source_binding = {
        str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    report = {
        "kind": "real_spatial_tools_v2_environment_image_smoke",
        "source_code_sha256": source_binding,
        "limits": {
            "max_source_images": 1,
            "max_objects": 2,
            "tool_timeout_seconds": args.tool_timeout,
            "total_timeout_seconds": args.total_timeout,
            "llm_api_calls": 0,
        },
        "source": {
            "environment_jsonl": str(args.environment_jsonl),
            "task_id": task.get("task_id"),
            "image_uri": str(image),
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        },
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename,
        "checks": {},
        "status": "running",
        "limitations": [
            "One source image; no benchmark accuracy claim",
            "No final LLM answer or model media-consumption check",
            "Successful geometry validates interface consistency, not ground-truth perception accuracy",
        ],
    }

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        temp = report_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False))
        temp.replace(report_path)

    def unavailable(name, reason):
        report["checks"][name] = {"status": "unavailable", "reason": reason}
        save()

    def run(name, tool_name, arguments, validator=None):
        remaining = args.total_timeout - (time.monotonic() - started)
        if remaining <= 0:
            report["checks"][name] = {
                "status": "failed",
                "error": "total diagnostic timeout exhausted",
            }
            save()
            return None
        request_path = args.output / f"{name}.request.json"
        result_path = args.output / f"{name}.result.json"
        request_path.write_text(
            json.dumps(
                {"tool_name": tool_name, "arguments": arguments}, allow_nan=False
            )
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(args.output),
            "--worker-request",
            str(request_path),
            "--worker-result",
            str(result_path),
        ]
        print(f"START {name}", flush=True)
        status = {
            "status": "failed",
            "request_path": str(request_path),
            "result_path": str(result_path),
        }
        try:
            with (
                (args.output / f"{name}.stdout.log").open("w") as stdout,
                (args.output / f"{name}.stderr.log").open("w") as stderr,
            ):
                process = subprocess.run(
                    command,
                    env={
                        **os.environ,
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1",
                    },
                    stdout=stdout,
                    stderr=stderr,
                    timeout=min(args.tool_timeout, remaining),
                    check=False,
                )
            status["exit_code"] = process.returncode
            if process.returncode:
                status["error"] = "worker_failed; inspect stderr log"
                result = None
            elif not result_path.is_file():
                status["error"] = "worker produced no ToolResult"
                result = None
            else:
                result = json.loads(result_path.read_text())
                status["tool_status"] = result["status"]
                if result["status"] != "succeeded":
                    status.update(
                        error=result.get("error_message"),
                        error_type=result.get("error_type"),
                    )
                    result = None
                else:
                    status["status"] = "passed"
                    if validator:
                        verdict, reason = validator(result)
                        status.update(status=verdict, reason=reason)
        except subprocess.TimeoutExpired:
            status["error"] = "tool timeout; worker killed"
            result = None
        except Exception as exc:  # noqa: BLE001 - each failure is persisted in the diagnostic report
            status.update(error_type=type(exc).__name__, error=str(exc))
            result = None
        report["checks"][name] = status
        save()
        print(f"DONE {name} {status['status']}", flush=True)
        return result if status["status"] == "passed" else None

    save()
    detection = run(
        "detect",
        "detect",
        {"image_uri": str(image), "queries": args.queries},
        lambda r: (
            ("passed", None)
            if r["structured_output"].get("detections")
            else ("unavailable", "no detections for configured source-image queries")
        ),
    )
    reconstruction = run(
        "reconstruct",
        "reconstruct",
        {"image_uris": [str(image)], "frame_indices": [0]},
        lambda r: (
            ("passed", None)
            if r["structured_output"].get("valid_point_count", 0)
            else ("unavailable", "no valid reconstructed points")
        ),
    )
    recon_uri = reconstruction["artifacts"][0]["uri"] if reconstruction else None
    centers, masks, selected = [], [], []
    mask_arrays = []
    if detection:
        candidates = sorted(
            detection["structured_output"]["detections"],
            key=lambda d: d.get("confidence", 0),
            reverse=True,
        )
        # Avoid treating two overlapping detections as two distinct objects.
        for candidate in candidates:
            a = candidate["bbox"]
            duplicate = False
            for existing in selected:
                b = existing["bbox"]
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                union = (
                    (a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1])
                    - intersection
                )
                duplicate |= union > 0 and intersection / union > 0.8
            if not duplicate:
                selected.append(candidate)
            if len(selected) == 2:
                break
        for index, candidate in enumerate(selected):
            segment = run(
                f"segment_{index}",
                "segment",
                {"image_uri": str(image), "boxes": [candidate["bbox"]]},
                lambda r: (
                    ("passed", None)
                    if r["structured_output"].get("mask_count", 0)
                    else ("unavailable", "segmentation returned no object mask")
                ),
            )
            if not segment:
                continue
            artifact = max(
                segment["artifacts"],
                key=lambda a: a.get("metadata", {}).get("score", 0),
            )
            import numpy as np
            from PIL import Image

            with Image.open(args.output / "storage" / artifact["uri"]) as binary:
                mask_array = np.asarray(binary.convert("L")) > 0
            repeated = False
            for previous in mask_arrays:
                union = int((previous | mask_array).sum())
                repeated |= (
                    union > 0 and int((previous & mask_array).sum()) / union > 0.8
                )
            if not mask_array.any() or repeated:
                report["checks"][f"segment_{index}"].update(
                    status="unavailable", reason="empty or duplicate object mask"
                )
                save()
                continue
            mask_arrays.append(mask_array)
            masks.append((index, artifact["uri"]))
            if recon_uri:
                center = run(
                    f"centroid_{index}",
                    "mask",
                    {
                        "operation": "centroid_3d",
                        "mask_uris": [artifact["uri"]],
                        "reconstruction_uri": recon_uri,
                        "frame_index": 0,
                        "source_image_uri": str(image),
                    },
                    lambda r: (
                        ("passed", None)
                        if r["structured_output"].get("available")
                        else ("unavailable", r["structured_output"].get("reason"))
                    ),
                )
                if center:
                    centers.append((index, center["structured_output"]))
    else:
        unavailable(
            "segmentation_chain",
            "detection prerequisite unavailable or failed; no invented boxes",
        )
    if len(centers) >= 2:
        first, second = centers[0][1], centers[1][1]
        run(
            "geometry_distance",
            "geometry",
            {
                "operation": "point_distance",
                "first": first["centroid"],
                "second": second["centroid"],
                "frame_id": first["frame_id"],
                "first_frame_id": first["frame_id"],
                "second_frame_id": second["frame_id"],
                "length_unit": first["length_unit"],
            },
            lambda r: (
                ("passed", None)
                if r["structured_output"].get("distance", -1) >= 0
                else ("failed", "invalid numerical distance")
            ),
        )
    else:
        unavailable(
            "geometry_distance",
            "need two distinct detected objects with valid 3D centroids",
        )
    if selected:
        run(
            "graph_relations",
            "graph",
            {
                "entities": [
                    {"id": str(i), "label": item["label"], "bbox": item["bbox"]}
                    for i, item in enumerate(selected)
                ]
            },
            lambda r: (
                ("passed", None)
                if r["structured_output"].get("edges")
                else ("unavailable", "need two objects to form relation edges")
            ),
        )
    else:
        unavailable("graph_relations", "no detected objects")
    if centers:
        run(
            "graph_plot",
            "graph",
            {
                "operation": "plot",
                "values": [value["centroid"][2] for _, value in centers],
                "x_label": "Detected object",
                "y_label": "World Z (reconstruction units)",
                "title": "Observed centroid coordinates",
            },
            lambda r: (
                ("passed", None)
                if any(a["artifact_type"] == "image" for a in r["artifacts"])
                else ("failed", "plot produced no image artifact")
            ),
        )
    else:
        unavailable("graph_plot", "no valid numerical centroid observations")
    if recon_uri:
        run(
            "scale_alignment",
            "scale",
            {
                "image_uri": str(image),
                "reconstruction_uri": recon_uri,
                "frame_index": 0,
                "resolution_level": 1,
            },
            lambda r: (
                ("passed", None)
                if r["structured_output"].get("alignment", {}).get("available")
                else (
                    "unavailable",
                    r["structured_output"]
                    .get("alignment", {})
                    .get("reason", "no alignment result"),
                )
            ),
        )
    else:
        unavailable(
            "scale_alignment", "reconstruction prerequisite unavailable or failed"
        )
    if recon_uri and masks:
        run(
            "pose_axes",
            "pose",
            {
                "operation": "object_frame",
                "image_uri": str(image),
                "mask_uri": masks[0][1],
                "reconstruction_uri": recon_uri,
                "frame_index": 0,
            },
            lambda r: (
                ("passed", None)
                if r["structured_output"].get("available")
                else ("unavailable", r["structured_output"].get("reason"))
            ),
        )
    else:
        unavailable("pose_axes", "requires reconstruction and valid object mask")
    core = (
        "detect",
        "reconstruct",
        "geometry_distance",
        "graph_relations",
        "graph_plot",
    )
    failed = any(c["status"] == "failed" for c in report["checks"].values())
    complete = all(
        report["checks"].get(name, {}).get("status") == "passed" for name in core
    )
    report["status"] = "failed" if failed else "passed" if complete else "unavailable"
    report["required_chain_complete"] = complete
    save()
    print(
        json.dumps({"status": report["status"], "report": str(report_path)}, indent=2),
        flush=True,
    )
    raise SystemExit(1 if failed else 0 if complete else 2)


if __name__ == "__main__":
    main()
