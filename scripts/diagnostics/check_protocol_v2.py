"""Bounded real-Qwen validation of v2 action spans, scores and LLM bank operations."""

import argparse
import gc
import os
from dataclasses import replace
from pathlib import Path

from spatialcraft.experiments.journal import RunJournal, digest
from spatialcraft.experiments.knowledge_generator import KnowledgeGenerator
from spatialcraft.experiments.runtime import ExperimentRuntime
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.experiments.usage import AuditedProviderV2
from spatialcraft.knowledge.skill import SeedCatalog
from spatialcraft.models import ContentPart, RequestBuilder, ToolDefinition
from spatialcraft.models.scoring import with_skill_prompt
from spatialcraft.storage.atomic_io import atomic_write_json, sha256_file


def main():
    import torch
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise SystemExit("Requires an allocated GPU")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = ExperimentSettings.load(
        args.project / "configs/experiments/qwen35_9b_spatialcraft_v2.yaml"
    )
    binding = {
        "kind": "real_qwen_v2_component_validation_not_benchmark",
        "job_id": os.environ["SLURM_JOB_ID"],
        "source_sha256": digest(
            {
                str(p.relative_to(args.project)): sha256_file(p)
                for p in (args.project / "src/spatialcraft").rglob("*.py")
            }
        ),
    }
    runtime = ExperimentRuntime(args.project, args.output, settings, binding)
    journal = RunJournal(args.output / "journal", binding)
    provider = AuditedProviderV2(runtime.local, journal, "executor")
    picture = args.output / "red.png"
    Image.new("RGB", (224, 224), "red").save(picture)
    request = (
        RequestBuilder(runtime.model)
        .system("Use instruct mode and output only a line 'Final Answer: <color>'.")
        .user(
            "What is the dominant color in this image?",
            media=(ContentPart.image_uri(str(picture)),),
        )
        .metadata(
            protocol_version="spatialcraft_v2",
            operation="execution.deployment",
            phase="diagnostic",
            chat_template_kwargs={"enable_thinking": False},
        )
        .settings(max_output_tokens=128, temperature=0, top_p=1, seed=42)
        .build()
    )
    request = with_skill_prompt(request, None)
    response = provider.generate(request)
    assert response.finish_reason == "stop", response.finish_reason
    action = response.raw["action_target"]
    assert action["status"] == "recorded", action
    score_request = replace(
        request,
        metadata={
            **request.metadata,
            "fixed_target_token_ids": action["token_ids"],
            "fixed_scoring_prefix_token_ids": action["prefix_token_ids"],
            "fixed_scoring_prefix": action["prefix"],
        },
    )
    old = provider.score(score_request, action["text"])
    candidate = SeedCatalog.pool().active()[0]
    new = provider.score(with_skill_prompt(score_request, candidate), action["text"])
    assert list(old.token_ids) == list(new.token_ids) == action["token_ids"]
    report = {
        **binding,
        "status": "running",
        "paid_api_called": False,
        "embedding_api_validated": False,
        "final_action": action,
        "original_token_logprobs": list(old.token_logprobs),
        "candidate_token_logprobs": list(new.token_logprobs),
        "scoring": "passed_same_recorded_tokens",
    }
    atomic_write_json(args.output / "report.json", report)
    print("V2_EXACT_ACTION_SCORING_PASSED", flush=True)
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
        RequestBuilder(runtime.model)
        .system("Call inspect_color once before answering.")
        .user(
            f"Call inspect_color with image_uri {picture}.",
            media=(ContentPart.image_uri(str(picture)),),
        )
        .tools((tool,))
        .tool_choice("auto", parallel=False)
        .metadata(
            protocol_version="spatialcraft_v2",
            operation="execution.accumulation",
            phase="diagnostic",
            chat_template_kwargs={"enable_thinking": False},
        )
        .settings(max_output_tokens=256, temperature=0, top_p=1, seed=42)
        .build()
    )
    tool_request = with_skill_prompt(tool_request, None)
    tool_response = provider.generate(tool_request)
    assert len(tool_response.tool_calls) == 1
    target = tool_response.raw["action_target"]
    assert target["status"] == "recorded", target
    tool_score = provider.score(
        replace(
            tool_request,
            metadata={
                **tool_request.metadata,
                "fixed_target_token_ids": target["token_ids"],
                "fixed_scoring_prefix_token_ids": target["prefix_token_ids"],
                "fixed_scoring_prefix": target["prefix"],
            },
        ),
        target["text"],
    )
    assert list(tool_score.token_ids) == target["token_ids"]
    report["native_tool_action"] = {
        "status": "passed",
        "target": target,
        "arguments": tool_response.tool_calls[0].arguments,
    }
    atomic_write_json(args.output / "report.json", report)
    print("V2_NATIVE_TOOL_TARGET_PASSED", flush=True)
    kb = AuditedProviderV2(runtime.local, journal, "knowledge_builder")
    generator = KnowledgeGenerator(
        settings,
        {"knowledge_builder": runtime.model},
        {"knowledge_builder": kb},
        args.project,
        journal,
    )
    generator.scope = {"phase": "diagnostic", "snapshot_id": "synthetic-bank-v1"}

    def check_merge(value):
        if value.get("decision") not in ("merge", "keep_separate") or not isinstance(
            value.get("reason"), str
        ):
            raise ValueError("Require decision and reason")
        if value["decision"] == "merge" and (
            value.get("source_refs") != ["E1@1"]
            or not value.get("condition")
            or not value.get("action")
        ):
            raise ValueError(
                "Merged item must use provided source and condition/action"
            )

    merge = generator(
        "experience.merge",
        {
            "incoming": {
                "reference": "E2@1",
                "condition": "For observer-relative directions",
                "action": "Identify observer axes before comparing.",
            },
            "candidates": [
                {
                    "reference": "E1@1",
                    "condition": "When directions are observer-relative",
                    "action": "Identify the observer axes first.",
                    "cosine": 0.99,
                }
            ],
            "max_words": 64,
            "expected_output": {
                "decision": "merge or keep_separate",
                "source_refs": ["E1@1"],
                "condition": "text if merge",
                "action": "text if merge",
                "reason": "text",
            },
        },
        validator=check_merge,
    )
    report["llm_merge"] = {"status": "passed", "decision": merge}
    atomic_write_json(args.output / "report.json", report)
    print("V2_LLM_MERGE_PASSED", flush=True)

    def check_manage(value):
        operations = value.get("operations")
        if not isinstance(operations, list) or len(operations) != 1:
            raise ValueError(
                "Return one merge operation reducing two duplicates to one"
            )
        op = operations[0]
        if (
            op.get("type") != "merge"
            or set(op.get("source_refs", [])) != {"E1@1", "E2@1"}
            or not op.get("condition")
            or not op.get("action")
            or not op.get("reason")
        ):
            raise ValueError(
                "Merge both supplied duplicates with condition/action and reason"
            )

    manage = generator(
        "experience.manage",
        {
            "capacity": 1,
            "experiences": [
                {
                    "reference": "E1@1",
                    "condition": "For observer-relative directions",
                    "action": "Identify observer axes first.",
                },
                {
                    "reference": "E2@1",
                    "condition": "For directions relative to an observer",
                    "action": "Establish observer axes before comparing.",
                },
            ],
            "instructions": "These two entries are equivalent. Reduce the full bank to capacity one by one supported merge.",
            "expected_output": {
                "operations": [
                    {
                        "type": "merge",
                        "source_refs": ["E1@1", "E2@1"],
                        "condition": "text",
                        "action": "text",
                        "reason": "text",
                    }
                ]
            },
        },
        validator=check_manage,
    )
    report["llm_manage"] = {"status": "passed", "decision": manage}
    report.update(
        status="passed",
        knowledge_modes=[
            {
                "operation": e["operation"],
                "mode": e["effective_reasoning_mode"],
                "max_output_tokens": e["max_output_tokens"],
                "status": e["status"],
            }
            for e in generator.audit
        ],
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
    )
    atomic_write_json(args.output / "report.json", report)
    print("V2_REAL_QWEN_COMPONENT_VALIDATION_PASSED", flush=True)
    runtime.local._model = None
    runtime.local._processor = None
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
