"""Allocated-GPU validation of native Qwen tools, seeded sampling and image scoring.

Never calls OpenAI or an embedding provider; no benchmark data is used.
"""

import argparse
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from spatialcraft.knowledge.skill import SeedCatalog
from spatialcraft.models import ContentPart, RequestBuilder, ToolDefinition
from spatialcraft.models.providers.transformers_local import TransformersLocalProvider
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.models.scoring import with_skill_prompt


def main():
    import torch
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Validate the 1024-token thinking protocol and conditional action scores",
    )
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise SystemExit("Requires an allocated GPU")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    config = ModelConfig.from_dict(load_yaml(root / "configs/models/qwen3.5-9b.yaml"))
    config = replace(
        config,
        local=replace(
            config.local,
            device_map="cuda:0",
            trust_remote_code=False,
            extra_load_kwargs={"local_files_only": True, "attn_implementation": "sdpa"},
        ),
    )
    provider = TransformersLocalProvider(config)
    tool = ToolDefinition(
        name="inspect_color",
        description="Inspect the color of the image",
        parameters={
            "type": "object",
            "properties": {"image_uri": {"type": "string"}},
            "required": ["image_uri"],
        },
    )
    with tempfile.TemporaryDirectory(prefix="spatialcraft-protocol-gpu-") as temp:
        picture = Path(temp) / "red.png"
        Image.new("RGB", (224, 224), "red").save(picture)
        builder = (
            RequestBuilder(config)
            .system("Follow the instruction.")
            .system("Visual evidence has priority.")
            .developer("Use the supplied skill conditionally.")
            .user(
                "What is the dominant color? Reply with one word.",
                media=(ContentPart.image_uri(str(picture)),),
            )
            .metadata(chat_template_kwargs={"enable_thinking": args.thinking})
            .settings(
                max_output_tokens=1024 if args.thinking else 32,
                temperature=0.7,
                top_p=0.9,
                seed=42,
            )
        )
        request = builder.build()
        first, second = provider.generate(request), provider.generate(request)
        assert (
            first.text == second.text
            and first.raw["generated_token_ids"] == second.raw["generated_token_ids"]
        )
        print("SEEDED_REPLAY_OK", first.text, flush=True)
        scoring_request = request
        if args.thinking:
            assert first.finish_reason == "stop" and first.usage.output_tokens <= 1024
            prefix = first.raw["sampled_thinking_prefix"]
            assert "</think>" in prefix
            scoring_request = replace(
                request,
                metadata={
                    **request.metadata,
                    "fixed_scoring_prefix": prefix,
                    "thinking_score_mode": "fixed_sampled_prefix",
                },
            )
            print(
                "THINKING_CLOSED_WITHIN_BUDGET", first.usage.output_tokens, flush=True
            )
        skill = SeedCatalog.pool().active()[0]
        old = provider.score(with_skill_prompt(scoring_request, skill), "red")
        changed = replace(
            skill,
            policy=(
                *skill.policy,
                "Inspect the image color carefully before answering.",
            ),
        )
        new = provider.score(with_skill_prompt(scoring_request, changed), "red")
        assert old.token_ids == new.token_ids
        assert all(
            torch.isfinite(torch.tensor(old.token_logprobs + new.token_logprobs))
        )
        print(
            "MULTIMODAL_FIXED_ACTION_SCORING_OK",
            old.token_ids,
            old.mean_logprob,
            new.mean_logprob,
            flush=True,
        )
        tool_request = (
            RequestBuilder(config)
            .system(
                "You must call inspect_color to inspect the image before answering."
            )
            .developer("Current procedure: call inspect_color now.")
            .user(
                f"Call inspect_color with image_uri {picture}.",
                media=(ContentPart.image_uri(str(picture)),),
            )
            .tools((tool,))
            .tool_choice("auto")
            .metadata(chat_template_kwargs={"enable_thinking": args.thinking})
            .settings(
                max_output_tokens=1024 if args.thinking else 256,
                temperature=0.7 if args.thinking else 0,
                top_p=0.9,
                seed=7,
            )
            .build()
        )
        response = provider.generate(tool_request)
        assert response.tool_calls and response.tool_calls[0].name == "inspect_color", (
            response
        )
        print("NATIVE_QWEN_TOOL_CALL_OK", response.tool_calls[0].arguments, flush=True)
        if args.thinking:
            assert (
                response.finish_reason == "stop"
                and "</think>" in response.raw["sampled_thinking_prefix"]
            )
            print(
                "THINKING_TOOL_CLOSED_WITHIN_BUDGET",
                response.usage.output_tokens,
                flush=True,
            )
    print("ALL_PROVIDER_PROTOCOL_CHECKS_OK_NO_API", flush=True)


if __name__ == "__main__":
    main()
