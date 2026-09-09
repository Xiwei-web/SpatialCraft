"""Validate actual FLA kernels, two-device generation/scoring, and prior OOM input."""

import argparse
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

from spatialcraft.experiments.fast_qwen import FastExperimentRuntime
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.knowledge.skill import SeedCatalog
from spatialcraft.models import ContentPart, RequestBuilder
from spatialcraft.models.scoring import with_skill_prompt
from spatialcraft.models.serialization import request_from_dict, response_to_dict
from spatialcraft.storage.atomic_io import atomic_write_json, read_json, sha256_file


def main():
    import torch
    from PIL import Image
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "paid_api_called": False}
    report["source_manifest_sha256"] = sha256_file(args.project / "source_manifest.json")

    def save():
        atomic_write_json(args.output / "acceptance.json", report)

    save()
    try:
        torch.set_num_threads(1)
        settings = ExperimentSettings.load(
            args.project / "configs/experiments/qwen35_9b_spatialcraft.yaml"
        )
        with tempfile.TemporaryDirectory(prefix="fla-qwen-validation-") as folder:
            runtime = FastExperimentRuntime(args.project, Path(folder), settings, {})
            provider, config = runtime.local, runtime.model
            print("LOAD_TWO_GPU_MODEL", flush=True)
            model, _ = provider._load()
            report["kernel_runtime"] = runtime.binding["kernel_runtime"]
            report["device_map"] = {k: str(v) for k, v in model.hf_device_map.items()}
            report["linear_kernel"] = qwen.chunk_gated_delta_rule.__module__
            # Compare the fused delta rule to the installed reference at small size.
            torch.manual_seed(42)
            q = torch.randn(1, 128, 4, 128, device="cuda:0", dtype=torch.bfloat16)
            k, v = torch.randn_like(q), torch.randn_like(q)
            g = -torch.rand(1, 128, 4, device="cuda:0")
            beta = torch.rand(1, 128, 4, device="cuda:0", dtype=torch.bfloat16)
            fast, _ = qwen.chunk_gated_delta_rule(
                q, k, v, g, beta, use_qk_l2norm_in_kernel=True, output_final_state=True
            )
            ref, _ = qwen.torch_chunk_gated_delta_rule(
                q, k, v, g, beta, use_qk_l2norm_in_kernel=True, output_final_state=True
            )
            error = (fast.float() - ref.float()).abs().max().item()
            torch.testing.assert_close(
                fast.float(), ref.float(), atol=0.025, rtol=0.025
            )
            report["delta_rule_max_abs_difference"] = error
            del q, k, v, fast, ref
            picture = Path(folder) / "red.png"
            Image.new("RGB", (224, 224), "red").save(picture)
            request = (
                RequestBuilder(config)
                .system("Answer directly without thinking.")
                .user(
                    "What is the dominant color? Reply with one word.",
                    media=(ContentPart.image_uri(str(picture)),),
                )
                .metadata(chat_template_kwargs={"enable_thinking": False})
                .settings(max_output_tokens=4096, temperature=0.7, top_p=0.9, seed=42)
                .build()
            )
            first, second = provider.generate(request), provider.generate(request)
            assert first.raw["generated_token_ids"] == second.raw["generated_token_ids"]
            assert "red" in first.text.lower() and first.finish_reason == "stop"
            report["seeded_multimodal_replay"] = True
            skill = SeedCatalog.pool().active()[0]
            changed = replace(
                skill, policy=(*skill.policy, "Check the visible image color.")
            )
            old = provider.score(with_skill_prompt(request, skill), "red")
            new = provider.score(with_skill_prompt(request, changed), "red")
            assert old.token_ids == new.token_ids and old.token_ids
            assert all(
                math.isfinite(x) for x in old.token_logprobs + new.token_logprobs
            )
            report["fixed_action_scoring"] = {
                "old": old.mean_logprob,
                "new": new.mean_logprob,
            }
            save()
            failed = (
                args.run
                / "robospatial/stages/tasks/00092/rollouts/00/execution/steps/0044/model/inputs.json"
            )
            long_request = request_from_dict(read_json(failed)["inputs"]["request"])
            report["oom_input_sha256"] = sha256_file(failed)
            for index in range(2):
                torch.cuda.reset_peak_memory_stats(index)
            print("REPLAY_EXACT_OOM_REQUEST_WITH_4096_BUDGET", flush=True)
            response = provider.generate(long_request)
            atomic_write_json(
                args.output / "oom-replay-response.json", response_to_dict(response)
            )
            report["oom_replay_usage"] = response_to_dict(response)["usage"]
            report["oom_replay_finish_reason"] = response.finish_reason
            report["peak_allocated_gib"] = [
                torch.cuda.max_memory_allocated(i) / 2**30 for i in range(2)
            ]
            assert response.text or response.tool_calls
            assert response.finish_reason != "length"
            print("SCORE_LONG_FIXED_ACTION", flush=True)
            score = provider.score(long_request, "yes")
            assert score.token_ids and all(
                math.isfinite(v) for v in score.token_logprobs
            )
            report["long_action_score"] = score.mean_logprob
            report["peak_allocated_gib_including_scoring"] = [
                torch.cuda.max_memory_allocated(i) / 2**30 for i in range(2)
            ]
            report["status"] = "passed"
            save()
            print(json.dumps(report), flush=True)
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
