import runpy
from pathlib import Path

import pytest

from spatialcraft.experiments.runtime import ExperimentRuntime
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.models.capabilities import Capability
from spatialcraft.models.registry import ProviderKind

ROOT = Path(__file__).resolve().parents[1]


def test_27b_retains_algorithm_and_uses_single_gpu_cpu_offload(tmp_path):
    original = ExperimentSettings.load(
        ROOT / "configs/experiments/qwen35_9b_spatialcraft.yaml"
    )
    larger = ExperimentSettings.load(
        ROOT / "configs/experiments/qwen36_27b_omni3d.yaml"
    )
    assert {
        key
        for key, value in original.to_dict().items()
        if larger.to_dict()[key] != value
    } == {"backbone"}
    runtime = ExperimentRuntime(ROOT, tmp_path, larger, {})
    assert runtime.model.provider is ProviderKind.TRANSFORMERS_LOCAL
    assert runtime.model.capabilities.supports(Capability.FIXED_TARGET_SCORING)
    assert runtime.model.local.dtype == "bfloat16"
    assert runtime.model.local.device_map == "auto"
    assert runtime.model.local.extra_load_kwargs["max_memory"] == {
        0: "22GiB",
        "cpu": "90GiB",
    }
    assert runtime.model.metadata["expected_gpus"] == 1
    assert runtime.model.metadata["allow_cpu_offload"] is True
    assert not runtime.model.local.load_in_4bit
    assert runtime.local._model is None  # Offline construction never loads weights.


def test_9b_keeps_single_gpu_default(tmp_path):
    runtime = ExperimentRuntime(ROOT, tmp_path, ExperimentSettings(), {})
    assert runtime.model.local.device_map == "cuda:0"
    assert "max_memory" not in runtime.model.local.extra_load_kwargs


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError, match="backbone"):
        ExperimentSettings(backbone="unknown")


def test_gpu_acceptance_allows_only_explicit_cpu_offload():
    validate = runpy.run_path(
        str(ROOT / "scripts/inference/check_experiment_provider.py")
    )["validate_device_map"]
    assert validate({"visual": 0, "layers": "cpu"}, 1, allow_cpu_offload=True)
    assert not validate({"": "cuda:0"}, 1)
    with pytest.raises(RuntimeError, match="placement"):
        validate({"visual": 0, "layers": "cpu"}, 1)
    for invalid in ({}, {"": "cpu"}, {"visual": 0, "layers": "disk"},
                    {"visual": 0, "layers": 1}, {"visual": 0, "layers": "meta"}):
        with pytest.raises(RuntimeError, match="placement"):
            validate(invalid, 1, allow_cpu_offload=True)


@pytest.mark.parametrize("backbone", ["qwen3.5-9b", "qwen3.6-27b-local"])
def test_offline_preflight_exercises_real_journal_without_inference(monkeypatch, backbone):
    from spatialcraft.experiments.run import check_runtime_initialization
    from spatialcraft.models.providers.openai_embeddings import OpenAIEmbeddingsProvider
    from spatialcraft.models.providers.transformers_local import (
        TransformersLocalProvider,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline startup attempted model/API inference")

    monkeypatch.setattr(TransformersLocalProvider, "_load", forbidden)
    monkeypatch.setattr(OpenAIEmbeddingsProvider, "embed", forbidden)
    settings = ExperimentSettings(backbone=backbone)
    assert check_runtime_initialization(ROOT, settings, {"code_sha256": "test"}, ("omni3d",)) == {
        "omni3d": "journal_create_commit_reopen_passed"
    }
