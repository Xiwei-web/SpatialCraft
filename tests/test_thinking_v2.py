from dataclasses import replace
from math import log
from types import SimpleNamespace

import pytest

from spatialcraft.models import (
    LocalModelConfig,
    ModelCapabilities,
    ModelConfig,
    ModelRequestError,
    ModelResponseError,
    ProviderKind,
    RequestBuilder,
)
from spatialcraft.models.providers.transformers_local import TransformersLocalProvider
from spatialcraft.models.response_parser import parse_generated_text
from spatialcraft.models.scoring import TargetLogprobScorer


def config(tmp_path):
    return ModelConfig(
        alias="local",
        model_id="local",
        provider=ProviderKind.TRANSFORMERS_LOCAL,
        local=LocalModelConfig(path=str(tmp_path)),
        capabilities=ModelCapabilities(
            supported=frozenset({"text_input", "fixed_target_scoring"})
        ),
    )


def test_fixed_thinking_is_context_not_scored_tokens(tmp_path, monkeypatch):
    import torch

    seen = []

    class UniformModel(torch.nn.Module):
        def forward(self, input_ids, use_cache=False, logits_to_keep=0):
            seen.append(input_ids.tolist())
            return SimpleNamespace(logits=torch.zeros((1, logits_to_keep, 128)))

    model = UniformModel()
    cfg = config(tmp_path)
    provider = TransformersLocalProvider(cfg, model=model, processor=object())
    monkeypatch.setattr(provider, "_render", lambda *args: ("PROMPT<think>\n", [], []))
    monkeypatch.setattr(
        provider,
        "_encode",
        lambda processor, text, images, videos: {
            "input_ids": torch.tensor([[ord(c) % 128 for c in text]])
        },
    )
    prefix = "Check the visible evidence.\n</think>\n\n"
    request = (
        RequestBuilder(cfg)
        .user("q")
        .metadata(
            chat_template_kwargs={"enable_thinking": True},
            fixed_scoring_prefix=prefix,
            thinking_score_mode="fixed_sampled_prefix",
        )
        .build()
    )
    score = TargetLogprobScorer(provider, model_alias="local").score(request, "red")
    assert score.token_ids == tuple(map(ord, "red"))
    assert score.token_logprobs == pytest.approx((-log(128),) * 3)
    assert score.prompt_token_count == len("PROMPT<think>\n" + prefix)
    assert score.metadata["thinking_tokens_excluded"] is True
    assert len(seen[0][0]) == score.prompt_token_count + 3
    with pytest.raises(ModelRequestError, match="completed fixed"):
        provider.score(
            replace(
                request, metadata={"chat_template_kwargs": {"enable_thinking": True}}
            ),
            "red",
        )


@pytest.mark.parametrize(
    "output,limit,expected",
    [
        ("Reason internally.</think>\n\nFinal Answer: red", 1024, "success"),
        ("Unclosed internal reasoning", 1024, "error"),
        ("Unclosed internal reasoning", 2, "truncated"),
    ],
)
def test_thinking_closure_and_total_generation_cap(
    tmp_path, monkeypatch, output, limit, expected
):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = config(tmp_path)
    generation = []

    def generate(**kwargs):
        generation.append(kwargs)
        return torch.tensor([[1, 2, 3, 4]])

    provider = TransformersLocalProvider(
        cfg,
        model=SimpleNamespace(generate=generate),
        processor=SimpleNamespace(batch_decode=lambda *args, **kwargs: [output]),
    )
    monkeypatch.setattr(
        provider, "_prepare", lambda *args: {"input_ids": torch.tensor([[1, 2]])}
    )
    request = (
        RequestBuilder(cfg)
        .user("q")
        .metadata(chat_template_kwargs={"enable_thinking": True})
        .settings(max_output_tokens=limit)
        .build()
    )
    if expected == "error":
        with pytest.raises(ModelResponseError, match="without closing"):
            provider.generate(request)
    else:
        response = provider.generate(request)
        if expected == "truncated":
            assert response.finish_reason == "length" and not response.tool_calls
        else:
            assert response.text == "Final Answer: red"
            assert (
                response.raw["sampled_thinking_prefix"]
                == "Reason internally.</think>\n\n"
            )
    assert generation[0]["max_new_tokens"] == limit


def test_unfinished_thinking_never_parses_as_an_action():
    with pytest.raises(ModelResponseError):
        parse_generated_text(
            "<think>Maybe call <tool_call>...", provider="test", model="test"
        )


def test_tokenizer_chat_eos_is_honored_even_at_budget_boundary(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    seen = {}

    def generate(**kwargs):
        seen.update(kwargs)
        return torch.tensor([[1, 2, 3, 248046]])

    cfg = config(tmp_path)
    provider = TransformersLocalProvider(
        cfg,
        model=SimpleNamespace(
            generate=generate, generation_config=SimpleNamespace(eos_token_id=248044)
        ),
        processor=SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token_id=248046, pad_token_id=248044),
            batch_decode=lambda *args, **kwargs: ["Brief reasoning.</think>\n\nyes"],
        ),
    )
    monkeypatch.setattr(
        provider, "_prepare", lambda *args: {"input_ids": torch.tensor([[1, 2]])}
    )
    response = provider.generate(
        RequestBuilder(cfg)
        .user("q")
        .metadata(chat_template_kwargs={"enable_thinking": True})
        .settings(max_output_tokens=2)
        .build()
    )
    assert seen["eos_token_id"] == [248044, 248046]
    assert seen["pad_token_id"] == 248044
    assert response.finish_reason == "stop" and response.text == "yes"
