"""Medium reasoning uses the same explicit budget for normal and recovery calls."""

from pathlib import Path

import pytest

from memp.runner import Settings, model_config, recovery_profiles
from spatialcraft.agent.decision import recovery_request
from spatialcraft.models import MessageRole, ModelMessage, ModelRequest
from spatialcraft.models.providers.openai_responses import OpenAIResponsesProvider

PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "gpt-5.4"])
def test_medium_normal_recovery_final_payload_omit_sampling_and_keep_budget(model):
    config = model_config(
        PROJECT,
        Settings(model=model, reasoning_effort="medium", max_output_tokens=16384),
    )
    request = ModelRequest(
        model_alias=config.alias,
        messages=(ModelMessage.text(MessageRole.USER, "Which object is left?"),),
        settings=config.generation,
        metadata={"recovery_profiles": recovery_profiles(config)},
    )
    provider = OpenAIResponsesProvider(config)
    for current in (
        request,
        recovery_request(request),
        recovery_request(request, forced_final=True),
    ):
        payload = provider.build_payload(current)
        assert payload["model"] == model
        assert payload["reasoning"] == {"effort": "medium"}
        assert payload["max_output_tokens"] == 16384
        assert not {"temperature", "top_p", "seed"}.intersection(payload)
        assert payload["store"] is False


def test_none_preserves_existing_short_recovery_budgets():
    config = model_config(PROJECT, Settings())
    assert recovery_profiles(config) == {
        "recovery": {"max_output_tokens": 1024},
        "forced_final": {"max_output_tokens": 512},
    }
