"""Three real environment-input provider calls with resumable stage commits.

NOT SpatialCraft evaluation: no agent/tools/knowledge/PPO/reward, reduced image
budget and short non-thinking generations. Never reads deployment or GT records.
Interrupting before ERQA and rerunning tests recovery after a durable model call.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from spatialcraft.experiments import RunJournal
from spatialcraft.experiments.prepare import DATASET_ORDER
from spatialcraft.models import ModelConfig, RequestBuilder
from spatialcraft.models.providers.transformers_local import TransformersLocalProvider
from spatialcraft.models.registry import load_yaml
from spatialcraft.schemas import TaskSample
from spatialcraft.storage.atomic_io import read_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--interrupt-before", choices=DATASET_ORDER)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    manifest = read_json(args.preparation / "manifest.json")
    if manifest["status"] != "prepared_not_run":
        raise ValueError("Expected a preparation manifest, not a benchmark result")
    cfg = ModelConfig.from_dict(load_yaml(project / "configs/models/qwen3.5-9b.yaml"))
    assert cfg.local is not None
    cfg = replace(
        cfg,
        local=replace(
            cfg.local,
            device_map="cuda:0",
            trust_remote_code=False,
            extra_load_kwargs={"local_files_only": True, "attn_implementation": "sdpa"},
        ),
    )
    settings = {
        "max_output_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "image_pixel_size": {"shortest_edge": 65536, "longest_edge": 262144},
        "enable_thinking": False,
        "selection": "first environment task for Robo/Omni; maximum-image environment task for ERQA",
    }
    journal = RunJournal(
        args.journal,
        {
            "purpose": "provider_smoke_only_not_spatialcraft_evaluation",
            "preparation_sha256": sha256_file(args.preparation / "manifest.json"),
            "script_sha256": sha256_file(Path(__file__)),
            "provider_sha256": sha256_file(
                project / "src/spatialcraft/models/providers/transformers_local.py"
            ),
            "model_config_sha256": sha256_file(
                project / "configs/models/qwen3.5-9b.yaml"
            ),
            "settings": settings,
            "versions": {
                name: version(name)
                for name in ("torch", "transformers", "qwen-vl-utils")
            },
        },
    )
    provider = None
    for name in DATASET_ORDER:
        relative = f"{name}/splits/environment.jsonl"
        path = args.preparation / relative
        if sha256_file(path) != manifest["input_files"][relative]["sha256"]:
            raise ValueError(f"Prepared split changed: {name}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        row = max(rows, key=lambda r: len(r["images"])) if name == "erqa" else rows[0]
        task = TaskSample.from_dict(row)
        if task.reference_answer is not None:
            raise ValueError("Smoke input must not include a reference answer")
        for image in task.images:
            if image.sha256 is None or sha256_file(image.uri) != image.sha256:
                raise ValueError("Input image checksum mismatch")

        def generate(name=name, task=task):
            nonlocal provider
            if args.interrupt_before == name:
                print(f"INTENTIONAL_INTERRUPTION_BEFORE {name}", flush=True)
                raise SystemExit(75)
            if provider is None:
                import torch

                if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
                    raise RuntimeError(
                        "Run real inference inside an allocated Slurm GPU job"
                    )
                if torch.cuda.mem_get_info()[0] < 24 * 1024**3:
                    raise RuntimeError(
                        "Need at least 24 GiB free GPU memory for this smoke"
                    )
                torch.set_num_threads(1)
                provider = TransformersLocalProvider(cfg)
                _, processor = provider._load()
                processor.image_processor.size = settings["image_pixel_size"]
            request = (
                RequestBuilder(cfg)
                .system(
                    "Answer the image question concisely. This is a connectivity test; no tools are available."
                )
                .task(task)
                .settings(max_output_tokens=64, temperature=0.0, top_p=1.0)
                .metadata(chat_template_kwargs={"enable_thinking": False})
                .build()
            )
            print(
                f"GENERATE {name} {task.task_id} images={len(task.images)}", flush=True
            )
            response = provider.generate(request)
            if not response.text and not response.tool_calls:
                raise RuntimeError("Empty provider response")
            return {
                "purpose": "provider_smoke_only_not_spatialcraft_evaluation",
                "task_id": task.task_id,
                "image_count": len(task.images),
                "request": asdict(request),
                "response": asdict(response),
                "reward_computed": False,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }

        result, reused = journal.execute(
            f"{name}/provider_generate", {"task": row}, generate
        )
        print(
            json.dumps(
                {
                    "dataset": name,
                    "reused": reused,
                    "images": result["image_count"],
                    "usage": result["response"]["usage"],
                    "answer": result["response"]["text"],
                }
            ),
            flush=True,
        )
    print("THREE_PROVIDER_SMOKES_COMMITTED_NOT_FORMAL_RESULTS", flush=True)


if __name__ == "__main__":
    main()
