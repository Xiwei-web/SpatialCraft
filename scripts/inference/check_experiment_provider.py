"""GPU acceptance for the configured experiment provider, without paid API calls."""

import argparse
import json
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from spatialcraft.experiments.runtime import ExperimentRuntime
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.knowledge.skill import SeedCatalog
from spatialcraft.models import ContentPart, RequestBuilder, ToolDefinition
from spatialcraft.models.scoring import with_skill_prompt


def validate_device_map(device_map, expected_gpus, *, allow_cpu_offload=False):
    placements = {str(value).removeprefix("cuda:") for value in device_map.values()}
    expected = {str(index) for index in range(expected_gpus)}
    allowed = expected | ({"cpu"} if allow_cpu_offload else set())
    if not placements or not placements <= allowed or not expected <= placements:
        raise RuntimeError(f"Unexpected model placement: {sorted(placements)}")
    return "cpu" in placements


def main():
    import torch
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise SystemExit("Requires a Slurm GPU allocation")
    torch.set_num_threads(1)
    settings = ExperimentSettings.load(args.config)
    report = {"backbone": settings.backbone, "paid_api_called": False}
    try:
        with tempfile.TemporaryDirectory(prefix="spatialcraft-27b-check-") as folder:
            runtime = ExperimentRuntime(args.project, Path(folder), settings, {})
            runtime.dataset("omni3d")
            report["runtime_initialization"] = "passed"
            provider, config = runtime.local, runtime.model
            expected = config.metadata.get("expected_gpus", 1)
            if torch.cuda.device_count() != expected:
                raise RuntimeError(f"Expected {expected} visible GPUs")
            model, _ = provider._load()
            device_map = getattr(model, "hf_device_map", {})
            report["device_map"] = {
                key: str(value) for key, value in device_map.items()
            }
            report["cpu_offload"] = validate_device_map(
                device_map,
                expected,
                allow_cpu_offload=config.metadata.get("allow_cpu_offload", False),
            )
            picture = Path(folder) / "red.png"
            Image.new("RGB", (224, 224), "red").save(picture)
            request = (
                RequestBuilder(config)
                .system("Answer directly without a thinking block.")
                .user(
                    "What is the dominant color? Reply with one word.",
                    media=(ContentPart.image_uri(str(picture)),),
                )
                .metadata(
                    chat_template_kwargs={"enable_thinking": settings.enable_thinking}
                )
                .settings(
                    max_output_tokens=settings.max_output_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    seed=42,
                )
                .build()
            )
            first, replay = provider.generate(request), provider.generate(request)
            assert first.finish_reason == "stop" and "red" in first.text.lower(), (
                first.text
            )
            assert first.raw["generated_token_ids"] == replay.raw["generated_token_ids"]
            report["image_answer"] = first.text
            report["seeded_replay"] = True
            skill = SeedCatalog.pool().active()[0]
            changed = replace(
                skill,
                policy=(*skill.policy, "Inspect the image color before answering."),
            )
            old = provider.score(with_skill_prompt(request, skill), "red")
            new = provider.score(with_skill_prompt(request, changed), "red")
            assert old.token_ids == new.token_ids and old.token_ids
            assert all(
                math.isfinite(v) for v in old.token_logprobs + new.token_logprobs
            )
            report["fixed_action_scoring"] = {
                "old": old.mean_logprob,
                "new": new.mean_logprob,
                "token_ids": old.token_ids,
            }
            tool = ToolDefinition(
                name="inspect_color",
                description="Inspect image color",
                parameters={
                    "type": "object",
                    "properties": {"image_uri": {"type": "string"}},
                    "required": ["image_uri"],
                },
            )
            tool_request = (
                RequestBuilder(config)
                .system("Call inspect_color before answering. Do not think aloud.")
                .user(
                    f"Call inspect_color with image_uri {picture}.",
                    media=(ContentPart.image_uri(str(picture)),),
                )
                .tools((tool,))
                .tool_choice("auto")
                .metadata(
                    chat_template_kwargs={"enable_thinking": settings.enable_thinking}
                )
                .settings(
                    max_output_tokens=settings.max_output_tokens, temperature=0, seed=7
                )
                .build()
            )
            response = provider.generate(tool_request)
            assert response.finish_reason == "stop" and response.tool_calls
            assert response.tool_calls[0].name == "inspect_color"
            report["native_tool_call"] = response.tool_calls[0].name
            report["gpu_memory"] = [
                {
                    "index": i,
                    "allocated": torch.cuda.memory_allocated(i),
                    "peak": torch.cuda.max_memory_allocated(i),
                }
                for i in range(expected)
            ]
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
