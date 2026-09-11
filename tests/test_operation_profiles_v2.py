from dataclasses import replace
from pathlib import Path

import pytest

from spatialcraft.agent.decision import recovery_request
from spatialcraft.experiments.knowledge_generator import KnowledgeGenerator
from spatialcraft.experiments.operation_profiles import (
    operation_profile,
    resolved_configuration,
)
from spatialcraft.experiments.reasoning_modes_v2 import (
    reasoning_controls,
    validate_operation_model,
)
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.models import (
    GenerationSettings,
    MessageRole,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
)
from spatialcraft.models.providers.gemini_native import GeminiNativeProvider
from spatialcraft.models.providers.openai_responses import OpenAIResponsesProvider
from spatialcraft.models.providers.vllm_client import VLLMClientProvider
from spatialcraft.models.registry import ModelConfig

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"experience_options": {"capactiy": 10}},
        {"experience_options": {"capacity": 10}},
        {"skill_options": {"batch_trajectories": 6}},
        {"skill_options": {"deduplication_mode": "typo"}},
        {"operations": {"experience.summary": {"token_budget": 123}}},
        {"operations": {"execution.accumulation": {"role": "knowledge_builder"}}},
        {"operations": {"execution.deployment": {"temperature": 0.7}}},
        {"operations": {"execution.fallback": {"temperature": 0.7}}},
        {"operations": {"experience.summary": {"repair_attempts": True}}},
        {"roles": {"scorer": "different-model"}},
        {"skill_options": {"preferred_low_reward": 2}},
    ],
)
def test_invalid_and_unused_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ExperimentSettings(protocol_version="spatialcraft_v2", **kwargs)


def test_candidate_budget_and_phase_fallback_resolve_to_actual_values():
    settings = ExperimentSettings(
        protocol_version="spatialcraft_v2",
        skill_generation_max_tokens=8192,
        operations={
            "execution.accumulation": {"temperature": 0.8},
            "execution.fallback": {"max_output_tokens": 2048},
        },
    )
    assert (
        operation_profile(settings, "skill.candidate_generation").max_output_tokens
        == 8192
    )
    assert operation_profile(settings, "execution.fallback").temperature == 0.8
    assert (
        operation_profile(settings, "execution.fallback", deployment=True).temperature
        == 0
    )
    assert (
        operation_profile(
            settings, "execution.fallback", deployment=True
        ).max_output_tokens
        == 2048
    )
    resolved = resolved_configuration(settings)
    assert resolved["effective_fallback_profiles"]["accumulation"]["top_p"] == 0.9
    assert resolved["skill_generation_max_tokens"] == 8192


def model(provider, mappings=None):
    value = {
        "alias": "remote",
        "provider": provider,
        "model_id": "configured-model",
        "capabilities": {"supported": ["text_input", "reasoning"]},
        "api": {"api_key_env": "UNUSED_TEST_KEY", "base_url": "http://localhost:1/v1"},
        "generation": {"reasoning_effort": "high"},
        "metadata": {"v2_reasoning_modes": mappings} if mappings else {},
    }
    return ModelConfig.from_dict(value)


class Capture(ModelProvider):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(
            provider="test", model=request.model_alias, text="{}", finish_reason="stop"
        )


def requests_for(model):
    capture = Capture()
    settings = ExperimentSettings(protocol_version="spatialcraft_v2")
    generator = KnowledgeGenerator(
        settings, {"knowledge_builder": model}, {"knowledge_builder": capture}, PROJECT
    )
    generator("experience.merge", {"seed": 123})
    generator("experience.summary", {"seed": 456})
    return capture.requests


def test_api_mode_mapping_required_and_responses_payload_contains_real_control():
    with pytest.raises(ValueError, match="v2_reasoning_modes"):
        reasoning_controls(model("openai_responses"), "instruct")
    configured = model(
        "openai_responses",
        {
            "instruct": {"reasoning_effort": "none"},
            "thinking": {"reasoning_effort": "medium"},
        },
    )
    instruct, thinking = requests_for(configured)
    provider = OpenAIResponsesProvider(configured)
    assert provider.build_payload(instruct)["reasoning"] == {"effort": "none"}
    assert provider.build_payload(thinking)["reasoning"] == {"effort": "medium"}
    assert instruct.settings.seed is None and thinking.settings.seed is None
    assert instruct.metadata["requested_generation_seed"] == 123
    assert thinking.metadata["requested_generation_seed"] == 456
    assert instruct.metadata["seed_applied"] is False
    assert (
        instruct.metadata["reasoning_control_validation"]
        == "configured_unverified_endpoint"
    )
    assert "chat_template_kwargs" not in instruct.metadata


def test_gemini_uses_native_thinking_config_and_clears_inherited_effort():
    configured = model(
        "gemini_native",
        {
            "instruct": {"thinking_config": {"thinking_budget": 0}},
            "thinking": {"thinking_config": {"thinking_budget": 2048}},
        },
    )
    instruct, thinking = requests_for(configured)
    provider = GeminiNativeProvider(configured)
    assert instruct.settings.reasoning_effort is None
    assert (
        provider.build_payload(instruct)["config"].thinking_config.thinking_budget == 0
    )
    assert (
        provider.build_payload(thinking)["config"].thinking_config.thinking_budget
        == 2048
    )
    bad = replace(
        configured,
        metadata={
            "v2_reasoning_modes": {
                "instruct": {"thinking_config": {"thinking_level": "MINIMAL"}},
                "thinking": {"thinking_config": {"thinking_level": "HIGH"}},
            }
        },
    )
    with pytest.raises(ValueError, match="thinking_budget=0"):
        reasoning_controls(bad, "instruct")


def test_vllm_mode_reaches_extra_body_instead_of_only_audit_metadata():
    configured = model(
        "vllm_client",
        {
            "instruct": {"chat_template_kwargs": {"enable_thinking": False}},
            "thinking": {"chat_template_kwargs": {"enable_thinking": True}},
        },
    )
    instruct, thinking = requests_for(configured)
    provider = VLLMClientProvider(configured)
    assert (
        provider.build_payload(instruct)["extra_body"]["chat_template_kwargs"][
            "enable_thinking"
        ]
        is False
    )
    assert (
        provider.build_payload(thinking)["extra_body"]["chat_template_kwargs"][
            "enable_thinking"
        ]
        is True
    )
    assert "reasoning_effort" not in provider.build_payload(thinking)


def test_remote_input_limit_is_not_silently_claimed_enforced():
    configured = model(
        "openai_responses",
        {
            "instruct": {"reasoning_effort": "none"},
            "thinking": {"reasoning_effort": "medium"},
        },
    )
    profile = replace(
        operation_profile(
            ExperimentSettings(protocol_version="spatialcraft_v2"), "experience.summary"
        ),
        max_input_tokens=4096,
    )
    with pytest.raises(ValueError, match="local tokenizer"):
        validate_operation_model(configured, profile)


@pytest.mark.parametrize("forced_final", [False, True])
def test_recovery_applies_its_own_input_budget_and_records_capped_output(forced_final):
    kind = "forced_final" if forced_final else "recovery"
    request = ModelRequest(
        model_alias="test",
        messages=(ModelMessage.text(MessageRole.USER, "Inspect the scene."),),
        settings=GenerationSettings(max_output_tokens=256),
        metadata={
            "protocol_version": "spatialcraft_v2",
            "operation_profile": {"max_input_tokens": 1000},
            "recovery_profiles": {
                kind: {"max_input_tokens": 2000, "max_output_tokens": 512}
            },
        },
    )
    recovered = recovery_request(request, forced_final=forced_final)
    assert recovered.metadata["operation_profile"]["max_input_tokens"] == 2000
    assert recovered.metadata["operation_profile"]["max_output_tokens"] == 256
    assert recovered.settings.max_output_tokens == 256
    assert recovered.metadata["operation"] == "execution." + kind
    assert request.metadata["operation_profile"]["max_input_tokens"] == 1000
