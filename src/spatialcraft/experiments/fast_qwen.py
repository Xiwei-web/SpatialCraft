"""Explicit two-GPU/FLA execution profile, isolated from other experiments."""

from dataclasses import asdict, replace
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from spatialcraft.models import SequenceScore
from spatialcraft.models.serialization import request_to_dict
from spatialcraft.storage.atomic_io import read_json, sha256_file

from .journal import digest
from .runtime import AuditedProvider, ConfiguredLocalProvider, ExperimentRuntime

OVERLAY = Path("/l/users/xiwei.liu/tool/env_overlays/qwen_fla_052")


class FastAuditedProvider(AuditedProvider):
    def score(self, request, target_text):
        # Old completed evolution stages are retained, but NEW gates must score
        # both parent/candidate under the same new backend, not reuse old-kernel
        # baseline probabilities. Generated trajectory commits remain immutable.
        inputs = {
            "request": request_to_dict(request, identity=False),
            "target": target_text,
            "execution_kernel": self.journal.binding["kernel_runtime"],
        }
        result, _ = self.journal.execute(
            "ppo_calls/" + digest(inputs),
            inputs,
            lambda: asdict(self.provider.score(request, target_text)),
        )
        return SequenceScore(**result)


def fast_local_config(local):
    return replace(
        local,
        device_map="balanced",
        extra_load_kwargs={
            **local.extra_load_kwargs,
            "max_memory": {0: "28GiB", 1: "34GiB"},
            "local_files_only": True,
            "attn_implementation": "sdpa",
        },
    )


@lru_cache(maxsize=1)
def kernel_identity():
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    if torch.cuda.device_count() != 2:
        raise RuntimeError("FLA profile requires exactly two allocated CUDA GPUs")
    if not qwen.is_fast_path_available:
        raise RuntimeError(
            "Qwen FLA/causal-conv1d kernels unavailable; refusing fallback"
        )
    manifest_path = OVERLAY / "installation_manifest.json"
    manifest = read_json(manifest_path)
    for name, checksum in manifest["files"].items():
        path = (OVERLAY / name).resolve()
        if not path.is_relative_to(OVERLAY.resolve()) or sha256_file(path) != checksum:
            raise RuntimeError("Kernel installation checksum changed")
    expected = {
        "flash-linear-attention": "0.5.2",
        "fla-core": "0.5.2",
        "causal-conv1d": "1.7.0",
    }
    if {name: version(name) for name in expected} != expected:
        raise RuntimeError("Unexpected kernel package versions")
    return {
        "profile": "qwen35_9b_fla_two_gpu_v1",
        "packages": expected,
        "installation_sha256": sha256_file(manifest_path),
        "cuda": torch.version.cuda,
        "expected_gpus": 2,
    }


class FastQwenProvider(ConfiguredLocalProvider):
    def _load(self):
        kernel_identity()
        model, processor = super()._load()
        placements = {
            str(v).removeprefix("cuda:") for v in model.hf_device_map.values()
        }
        if placements != {"0", "1"}:
            raise RuntimeError(
                f"Expected model on both GPUs without CPU/disk offload: {placements}"
            )
        layers = [m for m in model.modules() if hasattr(m, "chunk_gated_delta_rule")]
        if not layers or any(
            not m.chunk_gated_delta_rule.__module__.startswith("fla.") for m in layers
        ):
            raise RuntimeError(
                "Some Qwen linear-attention layers use the slow fallback"
            )
        return model, processor


class FastExperimentRuntime(ExperimentRuntime):
    def __init__(self, project, output, settings, binding):
        if settings.backbone != "qwen3.5-9b":
            raise ValueError("This execution profile is scoped to Qwen3.5-9B")
        super().__init__(project, output, settings, binding)
        self.model = replace(self.model, local=fast_local_config(self.model.local))
        self.local = FastQwenProvider(self.model, settings.image_max_pixels)
        self.binding = {**binding, "kernel_runtime": kernel_identity()}
