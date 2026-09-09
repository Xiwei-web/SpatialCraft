"""Compatibility for Qwen's direct convolution-weight access during CPU offload."""
from __future__ import annotations


def repair_qwen_cpu_offload(model) -> list[str]:
    """Keep tiny DeltaNet convolutions resident for both prefill and cached decode.

    Transformers 5.5.4 calls conv1d during prefill but reads conv1d.weight
    directly during decode. Accelerate's child forward hook is bypassed in
    the latter case, leaving the weight on meta. Pin only these small weights;
    large projections retain the configured CPU offload behavior.
    """
    if not hasattr(model, "named_modules"):
        return []
    repaired = []
    for name, module in model.named_modules():
        if type(module).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        conv = module.conv1d
        hook = getattr(conv, "_hf_hook", None)
        if hook is None or not getattr(hook, "offload", False):
            continue
        from accelerate.hooks import remove_hook_from_module
        from accelerate.utils import set_module_tensor_to_device

        device = hook.execution_device
        weights = {key: hook.weights_map[key] for key, _ in conv.named_parameters()}
        remove_hook_from_module(conv)
        for key, value in weights.items():
            set_module_tensor_to_device(conv, key, device, value=value)
        qualified_name = f"{name}.conv1d" if name else "conv1d"
        if hasattr(model, "hf_device_map"):
            model.hf_device_map[qualified_name] = device
        repaired.append(qualified_name)
    return repaired
