"""Real v2 Runtime diagnostic on environment-only RoboSpatial samples.

The launcher freezes src/configs/prompts and this diagnostic, then starts a new
Python process against that snapshot. No official deployment file is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


class DiagnosticInterrupted(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Keep progress/error reporting available before project modules import."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".diagnostic-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_report(path, fallback):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(fallback)


def error_record(error, stage):
    name = type(error).__name__
    lower = name.lower()
    if isinstance(error, DiagnosticInterrupted):
        category = "interrupted_or_time_limit"
    elif "outofmemory" in lower:
        category = "gpu_memory"
    elif "authentication" in lower or "permissiondenied" in lower:
        category = "credential_or_api_permission"
    elif "connection" in lower or "timeout" in lower or "ratelimit" in lower:
        category = "external_api_or_network"
    elif "validation" in lower or isinstance(error, (ValueError, AssertionError)):
        category = "protocol_or_output_validation"
    else:
        category = "runtime_or_infrastructure"
    # Deliberately omit exception text: API errors can contain credential excerpts.
    return {
        "error_type": f"{type(error).__module__}.{name}",
        "error_category": category,
        "failed_stage": stage,
        "traceback_frames": traceback.format_tb(error.__traceback__),
    }


def interrupt(signum, frame):
    raise DiagnosticInterrupted(f"received signal {signum}")


def install_signals():
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)


def frozen_launcher(project, output):
    helper_path = project / "scripts/inference/freeze_run_source.py"
    spec = importlib.util.spec_from_file_location(
        "diagnostic_source_freezer", helper_path
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    snapshot = helper.freeze(project.resolve(), (output / "code_snapshot").resolve())
    launchers = output / "launcher_code"
    launchers.mkdir(parents=True, exist_ok=True)
    for source in (Path(__file__).resolve(), helper_path):
        target = launchers / source.name
        if target.exists():
            if source.read_bytes() != target.read_bytes():
                raise ValueError(
                    "Diagnostic launcher changed; use a new output directory"
                )
        else:
            shutil.copy2(source, target)
    return snapshot, launchers / Path(__file__).name


def selected_environment(preparation, train_count):
    manifest_path = preparation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    relative = "robospatial/verification/environment.jsonl"
    source = preparation / relative
    # Verification data is required only for these completed environment tasks.
    if checksum(source) != manifest["input_files"][relative]["sha256"]:
        raise ValueError("Prepared environment labels changed")
    rows = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    if (
        len(rows) != manifest["datasets"]["robospatial"]["environment"]
        or len(rows) <= train_count
    ):
        raise ValueError("Environment sample count is missing or inconsistent")
    chosen = rows[: train_count + 1]
    if len({row["task_id"] for row in chosen}) != len(chosen):
        raise ValueError("Diagnostic tasks must be distinct")
    if any(
        row["dataset"] != "robospatial"
        or row["split"] != "train"
        or row.get("reference_answer") is None
        for row in chosen
    ):
        raise ValueError(
            "Diagnostic source must contain verified RoboSpatial environment tasks"
        )
    return chosen, {
        "preparation_manifest_sha256": checksum(manifest_path),
        "environment_input_sha256": checksum(source),
        "environment_source": str(source),
        "training_task_ids": [row["task_id"] for row in chosen[:-1]],
        "diagnostic_heldout_task_ids": [chosen[-1]["task_id"]],
    }


def worker(args):
    report_path = args.output / "report.json"
    report = read_report(report_path, {})
    started = time.monotonic()
    stage, runtime = "worker_initialization", None
    try:
        install_signals()
        import torch

        import spatialcraft
        from spatialcraft.experiments.journal import digest
        from spatialcraft.experiments.operation_profiles import resolved_configuration
        from spatialcraft.experiments.run import load_api_key_file
        from spatialcraft.experiments.runtime import ExperimentRuntime
        from spatialcraft.experiments.settings import ExperimentSettings
        from spatialcraft.experiments.usage import cost_report
        from spatialcraft.schemas import TaskSample, TaskSplit

        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise RuntimeError("Requires an allocated GPU")
        if (
            not Path(spatialcraft.__file__)
            .resolve()
            .is_relative_to(args.project / "src")
        ):
            raise RuntimeError("Worker did not import its frozen source snapshot")
        torch.set_num_threads(1)
        source_manifest = json.loads(
            (args.project / "source_manifest.json").read_text()
        )
        if any(
            checksum(args.project / relative) != expected
            for relative, expected in source_manifest.items()
        ):
            raise ValueError("Frozen source manifest failed verification")
        settings = replace(
            ExperimentSettings.load(
                args.project / "configs/experiments/qwen35_9b_spatialcraft_v2.yaml"
            ),
            max_steps=2,
            image_max_pixels=262144,
            experiment_name="diagnostic_real_pipeline",
        )
        if settings.ablations or settings.backbone != "qwen3.5-9b":
            raise ValueError("This diagnostic requires the full method with Qwen3.5-9B")
        stage = "environment_selection"
        raw, sources = selected_environment(args.preparation, args.train_tasks)
        training = tuple(TaskSample.from_dict(row) for row in raw[:-1])
        heldout_source = TaskSample.from_dict(raw[-1])
        heldout = (
            replace(
                heldout_source,
                split=TaskSplit.TEST,
                metadata={
                    **heldout_source.metadata,
                    "experiment_split": "deployment",
                    "diagnostic_source_split": "environment",
                    "diagnostic_heldout": True,
                },
            ),
        )
        for task in (*training, *heldout):
            for image in task.images:
                if image.sha256 is None or checksum(image.uri) != image.sha256:
                    raise ValueError("Diagnostic task image checksum changed")
        report.update(
            status="running",
            stage=stage,
            **sources,
            resolved_configuration=resolved_configuration(settings),
            expected_training_rollouts=len(training) * settings.rollouts_per_task,
            source_sha256=digest(source_manifest),
            imported_package=str(Path(spatialcraft.__file__).resolve()),
            registered_tools=[],
            executed_tool_names=[],
            official_deployment_read=False,
        )
        write_json(
            args.output / "resolved_configuration.json",
            report["resolved_configuration"],
        )
        write_json(report_path, report)
        stage = "credential_loading"
        load_api_key_file(args.api_key_file)
        stage = "runtime_initialization"
        binding = {
            "kind": "diagnostic_real_pipeline",
            "not_benchmark": True,
            "limited_horizon": True,
            "official_deployment_read": False,
            "source_heldout_split": "environment",
            "code_sha256": digest(source_manifest),
            "launcher_sha256": checksum(__file__),
            **sources,
            "settings": settings.to_dict(),
        }
        runtime = ExperimentRuntime(args.project, args.output, settings, binding)
        registered = sorted(
            definition.name for definition in runtime.tools.definitions()
        )
        if len(registered) != 11:
            raise ValueError("Expected all eleven real spatial tools")
        report.update(
            registered_tools=registered,
            stage=stage,
            model_roles="local Qwen3.5-9B executor/knowledge_builder/scorer",
            embedding_model=settings.embedding_model,
            paid_llm_called=False,
            embedding_api_enabled=True,
        )
        write_json(report_path, report)
        pipeline = runtime.dataset("robospatial")
        stage = "accumulation"
        report.update(stage=stage)
        write_json(report_path, report)
        print("V2_PIPELINE_ACCUMULATION_START", len(training), "tasks", flush=True)
        frozen = pipeline.accumulate(training)
        report.update(
            stage="environment_heldout_diagnostic",
            frozen_snapshot_id=frozen.snapshot_id,
            active_experiences=len(frozen.experiences.active()),
            active_skills=len(frozen.skills.active()),
        )
        write_json(report_path, report)
        stage = "environment_heldout_diagnostic"
        before = frozen.snapshot_id
        metrics = pipeline.deploy(heldout, frozen)
        if frozen.snapshot_id != before:
            raise AssertionError(
                "Diagnostic heldout execution mutated frozen knowledge"
            )
        report.update(metrics=metrics, frozen_read_only_check="passed")
        updates, rounds = [], []
        for index in range(len(training)):
            updates.append(
                pipeline.journal.read_committed(f"tasks/{index:05d}/experience_update")
            )
            rounds.append(
                pipeline.journal.read_committed(
                    f"evolution/{index:05d}/skill_evolution"
                )
            )
        rejected = [
            value.get("status")
            for value in updates
            if str(value.get("status", "")).startswith("skipped")
        ]
        report.update(
            status="completed_with_learning_rejections" if rejected else "passed",
            experience_update_statuses=[
                value.get("status", "unknown") for value in updates
            ],
            evolution_round_count=len(rounds),
            successful_training_and_heldout_execution=True,
            queue_batch_trigger_not_guaranteed=True,
            benchmark_accuracy_claimed=False,
        )
        write_json(
            args.output / "diagnostic_metrics.json",
            {
                "not_benchmark": True,
                "limited_horizon": True,
                "official_deployment_read": False,
                "source_split": "environment",
                "metrics": metrics,
            },
        )
        print("V2_REAL_PIPELINE_DIAGNOSTIC_COMPLETED", report["status"], flush=True)
        return 0
    except BaseException as error:  # noqa: BLE001 - diagnostic supervisor must record interruptions
        report.update(status="failed", **error_record(error, stage))
        print(
            "V2_REAL_PIPELINE_DIAGNOSTIC_FAILED",
            type(error).__name__,
            report["error_category"],
            flush=True,
        )
        return 1
    finally:
        report.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
        try:
            from spatialcraft.experiments.usage import cost_report

            ledger_root = args.output / "robospatial"
            if ledger_root.exists():
                report["usage"] = cost_report(
                    ledger_root, 1 if report.get("metrics") else None
                )
                write_json(ledger_root / "results/usage.json", report["usage"])
                operations = Counter()
                tools = set()
                for path in (ledger_root / "usage").glob("*.json"):
                    event = json.loads(path.read_text())
                    if event.get("actual_call") and event.get("operation"):
                        operations[event["operation"]] += 1
                    if event.get("kind") == "tool":
                        tools.add(event.get("operation"))
                report["observed_operation_calls"] = dict(operations)
                report["executed_tool_names"] = sorted(name for name in tools if name)
                report["manage_or_gate_not_exercised_is_not_a_failure"] = True
        except BaseException as error:  # noqa: BLE001 - diagnostic supervisor must record interruptions
            report["usage_report_error_type"] = type(error).__name__
        try:
            import torch

            report["peak_gpu_memory_bytes"] = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            )
        except BaseException as error:  # noqa: BLE001 - preserve the original diagnostic failure
            report["gpu_memory_report_error_type"] = type(error).__name__
        write_json(report_path, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--preparation",
        type=Path,
        default=Path(
            "/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1"
        ),
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("/home/xiwei.liu/.config/spatialcraft/openai_api_key"),
    )
    parser.add_argument("--train-tasks", type=int, choices=(1, 2), default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze source/launcher without GPU or API calls",
    )
    args = parser.parse_args(argv)
    args.output, args.project, args.preparation = (
        args.output.resolve(),
        args.project.resolve(),
        args.preparation.resolve(),
    )
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    base = {
        "kind": "diagnostic_real_pipeline",
        "not_benchmark": True,
        "limited_horizon": True,
        "official_deployment_read": False,
        "source_heldout_split": "environment",
        "status": "running",
        "started_at": now(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "stage": "source_freeze",
        "training_tasks": args.train_tasks,
        "rollouts_per_task": 4,
        "heldout_environment_tasks": 1,
        "max_environment_steps": 2,
        "image_max_pixels": 262144,
        "method_ablations": [],
        "source_project": str(args.project),
        "output": str(args.output),
    }
    write_json(report_path, base)
    process = None
    stage = "source_freeze"
    try:
        install_signals()
        snapshot, launcher = frozen_launcher(args.project, args.output)
        base.update(
            source_snapshot=str(snapshot),
            source_manifest_sha256=checksum(snapshot / "source_manifest.json"),
            launcher_sha256=checksum(launcher),
        )
        if args.prepare_only:
            base.update(status="prepared_not_run", finished_at=now(), stage="prepared")
            write_json(report_path, base)
            print("V2_PIPELINE_SOURCE_PREPARED", snapshot, flush=True)
            return 0
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Execution requires a Slurm GPU allocation")
        stage = "worker_execution"
        base.update(stage=stage)
        write_json(report_path, base)
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=str(snapshot / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="1",
        )
        command = [
            sys.executable,
            str(launcher),
            "--worker",
            "--project",
            str(snapshot),
            "--output",
            str(args.output),
            "--preparation",
            str(args.preparation),
            "--api-key-file",
            str(args.api_key_file),
            "--train-tasks",
            str(args.train_tasks),
        ]
        process = subprocess.Popen(command, env=environment, cwd=snapshot)
        code = process.wait()
        report = read_report(report_path, base)
        report["worker_returncode"] = code
        if code and report.get("status") == "running":
            report.update(
                status="failed",
                error_category="worker_exit_without_final_report",
                failed_stage=stage,
            )
        write_json(report_path, report)
        return code
    except BaseException as error:  # noqa: BLE001 - diagnostic supervisor must record interruptions
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        report = read_report(report_path, base)
        report.update(status="failed", finished_at=now(), **error_record(error, stage))
        write_json(report_path, report)
        print(
            "V2_PIPELINE_LAUNCHER_FAILED",
            type(error).__name__,
            report["error_category"],
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
