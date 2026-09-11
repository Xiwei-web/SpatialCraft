"""Explicit provider controls; configured support is not endpoint validation."""

from copy import deepcopy

from spatialcraft.models.capabilities import Capability
from spatialcraft.models.registry import ProviderKind


def reasoning_controls(model, mode):
    if mode not in {"instruct", "thinking"}:
        raise ValueError("Unknown reasoning mode")
    if mode == "thinking":
        model.capabilities.require(Capability.REASONING)
    extra = deepcopy(model.generation.extra)
    if model.provider is ProviderKind.TRANSFORMERS_LOCAL:
        return {"reasoning_effort": None, "extra": extra}, {
            "chat_template_kwargs": {"enable_thinking": mode == "thinking"},
            "reasoning_control": "local_chat_template_enable_thinking",
            "reasoning_control_validation": "local_template_adapter",
        }
    mappings = model.metadata.get("v2_reasoning_modes")
    if not isinstance(mappings, dict) or set(mappings) != {"instruct", "thinking"}:
        raise ValueError(
            f"{model.alias}: API/vLLM v2 requires explicit metadata.v2_reasoning_modes for instruct and thinking"
        )
    control = deepcopy(mappings[mode])
    if not isinstance(control, dict):
        raise ValueError("Reasoning mode control must be an object")  # noqa: TRY004 - configuration validation
    metadata = {
        "reasoning_control_validation": "configured_unverified_endpoint",
        "reasoning_control": control,
    }
    provider = model.provider
    if provider in {ProviderKind.OPENAI_RESPONSES, ProviderKind.OPENAI_CHAT_COMPATIBLE}:
        if set(control) != {"reasoning_effort"} or not isinstance(
            control["reasoning_effort"], str
        ):
            raise ValueError(
                "Responses/chat mode mappings require only reasoning_effort"
            )
        effort = control["reasoning_effort"]
        if (mode == "instruct" and effort != "none") or (
            mode == "thinking"
            and effort not in {"minimal", "low", "medium", "high", "xhigh"}
        ):
            raise ValueError(
                "Instruct requires explicit none; Thinking requires a nonzero reasoning effort"
            )
        if {"reasoning", "reasoning_effort"} & set(extra):
            raise ValueError(
                "Model generation.extra conflicts with operation reasoning controls"
            )
        return {"reasoning_effort": effort, "extra": extra}, metadata
    if provider is ProviderKind.GEMINI_NATIVE:
        if set(control) != {"thinking_config"} or not isinstance(
            control["thinking_config"], dict
        ):
            raise ValueError("Gemini mode mappings require thinking_config")
        config = control["thinking_config"]
        allowed = {"thinking_budget", "thinking_level", "include_thoughts"}
        if set(config) - allowed or ("thinking_budget" in config) == (
            "thinking_level" in config
        ):
            raise ValueError(
                "Declare exactly one Gemini thinking_budget or thinking_level"
            )
        budget = config.get("thinking_budget")
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError("Gemini thinking_budget must be a nonnegative integer")
        if mode == "instruct" and budget != 0:
            raise ValueError(
                "Gemini Instruct requires an endpoint-declared thinking_budget=0; minimal thinking is not disabled thinking"
            )
        if mode == "thinking" and (
            budget == 0
            or (budget is None and not isinstance(config.get("thinking_level"), str))
        ):
            raise ValueError(
                "Gemini Thinking requires positive budget or explicit thinking_level"
            )
        if (
            "include_thoughts" in config
            and type(config["include_thoughts"]) is not bool
        ):
            raise ValueError("Gemini include_thoughts must be boolean")
        if "thinking_config" in extra:
            raise ValueError(
                "Model generation.extra conflicts with Gemini thinking config"
            )
        extra["thinking_config"] = config
        return {"reasoning_effort": None, "extra": extra}, metadata
    if provider is ProviderKind.VLLM_CLIENT:
        if (
            set(control) != {"chat_template_kwargs"}
            or not isinstance(control["chat_template_kwargs"], dict)
            or type(control["chat_template_kwargs"].get("enable_thinking")) is not bool
            or control["chat_template_kwargs"]
            != {"enable_thinking": mode == "thinking"}
        ):
            raise ValueError(
                "vLLM mappings must explicitly toggle chat_template_kwargs.enable_thinking"
            )
        body = extra.setdefault("extra_body", {})
        if not isinstance(body, dict) or "chat_template_kwargs" in body:
            raise ValueError("Model extra_body conflicts with vLLM reasoning mapping")
        body["chat_template_kwargs"] = control["chat_template_kwargs"]
        return {"reasoning_effort": None, "extra": extra}, metadata
    raise ValueError(f"No explicit v2 reasoning adapter for {provider.value}")


def validate_operation_model(model, profile):
    for mode in (
        {profile.reasoning_mode, "instruct"}
        if profile.repair_attempts
        else {profile.reasoning_mode}
    ):
        reasoning_controls(model, mode)
    cap = model.capabilities.max_output_tokens
    maximum = max(
        profile.max_output_tokens,
        profile.repair_max_tokens if profile.repair_attempts else 0,
    )
    if cap is not None and maximum > cap:
        raise ValueError(
            f"Operation output/repair budget exceeds {model.alias} capability"
        )
    if (
        profile.max_input_tokens is not None
        and model.provider is not ProviderKind.TRANSFORMERS_LOCAL
    ):
        raise ValueError(
            "Exact input budget enforcement is currently supported only by the local tokenizer adapter"
        )
