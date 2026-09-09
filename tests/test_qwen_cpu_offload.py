"""Real Qwen/Accelerate regression for direct convolution-weight reads."""
from copy import deepcopy

import pytest
import torch
from accelerate import cpu_offload
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from spatialcraft.models.providers.offload import repair_qwen_cpu_offload


def test_offloaded_qwen_decode_matches_resident_weights():
    torch.manual_seed(24)
    config = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_value_head_dim=8,
        linear_key_head_dim=8,
        layer_types=["linear_attention"],
    )
    resident = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    offloaded = deepcopy(resident)
    cpu_offload(offloaded, execution_device=torch.device("cpu"))
    prompt = torch.randn(1, 5, 32)
    continuation = torch.randn(1, 1, 32)
    cache = DynamicCache(config=config)
    update = offloaded.causal_conv1d_update

    def detect_missing_weight(*args, **kwargs):
        assert args[2].device.type != "meta", "decode bypassed the weight loading hook"
        return update(*args, **kwargs)

    offloaded.causal_conv1d_update = detect_missing_weight
    with torch.inference_mode():
        offloaded(prompt, cache_params=cache)
        with pytest.raises(AssertionError, match="bypassed"):
            offloaded(continuation, cache_params=cache)
    offloaded.causal_conv1d_update = update
    assert repair_qwen_cpu_offload(offloaded) == ["conv1d"]
    assert repair_qwen_cpu_offload(offloaded) == []  # Safe to load repeatedly.
    assert offloaded.conv1d.weight.device.type == "cpu"
    assert offloaded.in_proj_qkv.weight.device.type == "meta"

    with torch.inference_mode():
        for _ in range(2):
            expected_cache, actual_cache = DynamicCache(config=config), DynamicCache(config=config)
            for hidden in (prompt, continuation, continuation):
                expected = resident(hidden, cache_params=expected_cache)
                actual = offloaded(hidden, cache_params=actual_cache)
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Fixed-action scoring uses a fresh full-prefix forward without cache.
            torch.testing.assert_close(offloaded(prompt), resident(prompt), rtol=0, atol=0)
