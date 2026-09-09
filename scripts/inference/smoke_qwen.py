"""Offline Qwen processor checks and optional single-GPU inference smoke test.

Run GPU modes inside a Slurm GPU allocation, never on the login node.
This tests the inference environment, not SpatialCraft's provider implementation.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("9b", "27b"), default="9b")
    parser.add_argument(
        "--backend", choices=("check", "transformers", "vllm"), default="check"
    )
    parser.add_argument(
        "--model-root", type=Path, default=Path("/l/users/xiwei.liu/model")
    )
    args = parser.parse_args()
    name = "Qwen3.5-9B" if args.model == "9b" else "Qwen3.6-27B"
    path = args.model_root / name
    if not path.is_dir():
        raise SystemExit(f"Missing local model: {path}")

    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor

    print(
        json.dumps(
            {p: version(p) for p in ("torch", "transformers", "vllm", "accelerate")}
        ),
        flush=True,
    )
    config = AutoConfig.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    processor = AutoProcessor.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    picture = Image.new("RGB", (224, 224), (255, 0, 0))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": picture},
                {
                    "type": "text",
                    "text": "What is the main color of this image? Answer with one color word.",
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=[prompt], images=[picture], return_tensors="pt")
    print(
        json.dumps(
            {
                "model": name,
                "architecture": config.architectures,
                "input_shapes": {
                    k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")
                },
            }
        ),
        flush=True,
    )
    if args.backend == "check":
        print("PROCESSOR_CHECK_OK", flush=True)
        return
    if not torch.cuda.is_available():
        raise SystemExit("GPU mode requires an allocated CUDA GPU.")
    free, total = torch.cuda.mem_get_info()
    minimum = (24 if args.model == "9b" else 65) * 1024**3
    if free < minimum:
        raise SystemExit(
            f"Insufficient free VRAM for this single-GPU smoke test: {free / 1024**3:.1f} GiB; need >= {minimum / 1024**3:.0f} GiB."
        )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "cuda": torch.version.cuda,
                "free_bytes": free,
                "total_bytes": total,
            }
        ),
        flush=True,
    )
    if args.backend == "transformers":
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        model.eval()
        inputs = inputs.to("cuda:0")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        answer = processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
    else:
        from vllm import LLM, SamplingParams

        model = LLM(
            model=str(path),
            tokenizer=str(path),
            dtype="bfloat16",
            trust_remote_code=False,
            max_model_len=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.80,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1, "video": 0},
        )
        output = model.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": picture}}],
            SamplingParams(max_tokens=32, temperature=0),
        )
        answer = output[0].outputs[0].text
    print(
        json.dumps({"backend": args.backend, "model": name, "answer": answer}),
        flush=True,
    )
    if not answer.strip():
        raise SystemExit("Model returned empty output.")
    print("INFERENCE_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
