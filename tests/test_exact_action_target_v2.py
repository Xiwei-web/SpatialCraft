from math import log
from types import SimpleNamespace

import pytest

from spatialcraft.agent.action_target import capture_action_target
from spatialcraft.models import (
    LocalModelConfig,
    ModelCapabilities,
    ModelConfig,
    ModelRequestError,
    ProviderKind,
    RequestBuilder,
)
from spatialcraft.models.providers.transformers_local import TransformersLocalProvider


class PieceTokenizer:
    name_or_path = "fixed-piece-test"

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[i] for i in ids)


def test_capture_preserves_original_whitespace_token_boundaries_and_prefix():
    pieces = {1: "reason</think>", 2: "\nFinal", 3: " Answer: ", 4: "A", 5: ""}
    tokenizer = PieceTokenizer(pieces)
    text = tokenizer.decode([1, 2, 3, 4, 5])
    target = capture_action_target(text, [1, 2, 3, 4, 5], tokenizer)
    assert target["status"] == "recorded"
    assert target["token_ids"] == [2, 3, 4]
    assert target["text"] == "\nFinal Answer: A"
    assert target["prefix_token_ids"] == [1]
    assert target["prefix"] == "reason</think>"


def test_capture_rejects_marker_embedded_with_explanation_in_same_token():
    tokenizer = PieceTokenizer({1: "Explanation\nFinal Answer:", 2: " A"})
    result = capture_action_target(tokenizer.decode([1, 2]), [1, 2], tokenizer)
    assert result == {
        "status": "unavailable",
        "reason": "action_boundary_not_token_aligned",
    }


@pytest.mark.parametrize(
    "text",
    [
        "A",
        "  A  ",
        "Brief explanation\nA",
        '{"final_answer": "A"}',
        "<tool_call><function=detect></function></tool_call>",
    ],
)
def test_bare_final_json_and_tool_spans_are_recorded(text):
    tokenizer = PieceTokenizer(dict(enumerate(text)))
    result = capture_action_target(text, list(range(len(text))), tokenizer)
    assert result["status"] == "recorded"
    assert tokenizer.decode(result["token_ids"]) == result["text"]


@pytest.mark.parametrize(
    "text", ["", '{"final_answer":', "[", "<tool_call><function=detect>"]
)
def test_incomplete_structures_do_not_become_plain_final_spans(text):
    tokenizer = PieceTokenizer(dict(enumerate(text)))
    assert (
        capture_action_target(text, list(range(len(text))), tokenizer)["status"]
        == "unavailable"
    )


def test_capture_uses_same_last_thinking_closure_as_parser():
    text = "first</think>\nFinal Answer: B\nsecond</think>\nFinal Answer: A"
    tokenizer = PieceTokenizer(dict(enumerate(text)))
    result = capture_action_target(text, list(range(len(text))), tokenizer)
    assert result["text"].strip() == "Final Answer: A"
    assert "Final Answer: B" in result["prefix"]


def provider_fixture(tmp_path, monkeypatch, *, limited=False):
    import torch

    seen = []
    tokenizer = PieceTokenizer({4: "context", 5: " ", 7: "Final Answer:", 8: " A"})

    def output(input_ids, count):
        seen.append(input_ids.tolist())
        logits = torch.zeros((1, input_ids.shape[-1], 16))
        logits[0, -3, 7] = 2
        logits[0, -2, 8] = 3
        return SimpleNamespace(logits=logits[:, -count:, :] if count else logits)

    class FullModel(torch.nn.Module):
        def forward(
            self, input_ids, attention_mask=None, token_type_ids=None, use_cache=False
        ):
            assert input_ids.shape == attention_mask.shape == token_type_ids.shape
            return output(input_ids, 0)

    class LimitedModel(torch.nn.Module):
        def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            use_cache=False,
            logits_to_keep=0,
        ):
            assert logits_to_keep == 3
            return output(input_ids, logits_to_keep)

    cfg = ModelConfig(
        alias="local",
        model_id="local",
        provider=ProviderKind.TRANSFORMERS_LOCAL,
        local=LocalModelConfig(path=str(tmp_path)),
        capabilities=ModelCapabilities(
            supported=frozenset({"text_input", "fixed_target_scoring"})
        ),
    )
    provider = TransformersLocalProvider(
        cfg, model=LimitedModel() if limited else FullModel(), processor=tokenizer
    )
    monkeypatch.setattr(provider, "_render", lambda *args: ("PROMPT", [], []))
    encodes = []

    def encode(*args):
        encodes.append(args[1])
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "token_type_ids": torch.tensor([[0, 0]]),
        }

    monkeypatch.setattr(provider, "_encode", encode)
    request = (
        RequestBuilder(cfg)
        .user("q")
        .metadata(
            protocol_version="spatialcraft_v2",
            fixed_target_token_ids=[7, 8],
            fixed_scoring_prefix_token_ids=[4, 5],
            fixed_scoring_prefix="context ",
        )
        .build()
    )
    return provider, request, seen, encodes


@pytest.mark.parametrize("limited", [False, True])
def test_fixed_token_scoring_preserves_ids_and_excludes_prefix(
    tmp_path, monkeypatch, limited
):
    import torch

    provider, request, seen, encodes = provider_fixture(
        tmp_path, monkeypatch, limited=limited
    )
    result = provider.score(request, "Final Answer: A")
    assert seen == [[[1, 2, 4, 5, 7, 8]]]
    assert encodes == ["PROMPT"]
    assert result.token_ids == (7, 8) and result.prompt_token_count == 4
    expected = [
        float(torch.log_softmax(torch.tensor([value] + [0.0] * 15), 0)[0])
        for value in (2.0, 3.0)
    ]
    assert result.token_logprobs == pytest.approx(expected)
    assert result.token_logprobs != pytest.approx((-log(16),) * 2)


def test_fixed_token_scoring_rejects_target_text_mismatch(tmp_path, monkeypatch):
    provider, request, _, _ = provider_fixture(tmp_path, monkeypatch)
    with pytest.raises(ModelRequestError, match="exact token IDs"):
        provider.score(request, "changed action")


def test_qwen35_multimodal_type_ids_extend_text_and_preserve_real_mrope(
    tmp_path, monkeypatch
):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    provider, request, _, _ = provider_fixture(tmp_path, monkeypatch)
    base_ids = torch.tensor([[1, 2, 9, 9, 9, 9, 3]])
    base_types = torch.tensor([[0, 0, 1, 1, 1, 1, 0]])
    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.tensor([[11.0, 12.0, 13.0]])
    seen = []

    class MropeModel(torch.nn.Module):
        config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        get_vision_position_ids = Qwen3_5Model.get_vision_position_ids

        def forward(
            self,
            input_ids,
            attention_mask,
            mm_token_type_ids,
            image_grid_thw,
            pixel_values,
            use_cache=False,
        ):
            assert (
                input_ids.shape
                == attention_mask.shape
                == mm_token_type_ids.shape
                == (1, 11)
            )
            assert torch.equal(mm_token_type_ids[:, :7], base_types)
            assert mm_token_type_ids[:, 7:].tolist() == [[0, 0, 0, 0]]
            assert torch.equal(image_grid_thw, grid) and torch.equal(
                pixel_values, pixels
            )
            original_positions, _ = Qwen3_5Model.get_rope_index(
                self,
                base_ids,
                base_types,
                image_grid_thw=grid,
                attention_mask=torch.ones_like(base_ids),
            )
            positions, _ = Qwen3_5Model.get_rope_index(
                self,
                input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            assert torch.equal(positions[:, :, :7], original_positions)
            assert torch.equal(positions[0, :, 7:], positions[1, :, 7:])
            assert torch.equal(positions[1, :, 7:], positions[2, :, 7:])
            seen.append(positions)
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[-1], 16)))

    provider._model = MropeModel()
    monkeypatch.setattr(
        provider,
        "_load",
        lambda: (
            provider._model,
            PieceTokenizer({4: "context", 5: " ", 7: "Final Answer:", 8: " A"}),
        ),
    )
    monkeypatch.setattr(
        provider,
        "_encode",
        lambda *args: {
            "input_ids": base_ids.clone(),
            "attention_mask": torch.ones_like(base_ids),
            "mm_token_type_ids": base_types.clone(),
            "image_grid_thw": grid.clone(),
            "pixel_values": pixels.clone(),
        },
    )
    result = provider.score(request, "Final Answer: A")
    assert result.token_ids == (7, 8) and result.prompt_token_count == 9
    assert len(seen) == 1


def test_incomplete_knowledge_thinking_keeps_usage_and_receives_bounded_repair(
    tmp_path, monkeypatch
):
    from pathlib import Path

    import torch

    from spatialcraft.experiments.knowledge_generator import KnowledgeGenerator
    from spatialcraft.experiments.settings import ExperimentSettings

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    outputs = ["unfinished analysis without closure", "{}"]
    invocations, responses = [], []
    config = ModelConfig(
        alias="local",
        model_id="local",
        provider=ProviderKind.TRANSFORMERS_LOCAL,
        local=LocalModelConfig(path=str(tmp_path)),
        capabilities=ModelCapabilities(
            supported=frozenset({"text_input", "reasoning", "fixed_target_scoring"})
        ),
    )

    def generate(**kwargs):
        invocations.append(kwargs)
        return torch.tensor([[1, 2, 3, 4]])

    processor = SimpleNamespace(
        batch_decode=lambda *args, **kwargs: [outputs[len(invocations) - 1]]
    )
    provider = TransformersLocalProvider(
        config,
        model=SimpleNamespace(
            generate=generate, config=SimpleNamespace(max_position_embeddings=100000)
        ),
        processor=processor,
    )
    monkeypatch.setattr(
        provider, "_prepare", lambda *args: {"input_ids": torch.tensor([[1, 2]])}
    )

    class RecordingProvider:
        def generate(self, request):
            result = provider.generate(request)
            responses.append(result)
            return result

    builder = KnowledgeGenerator(
        ExperimentSettings(protocol_version="spatialcraft_v2"),
        {"knowledge_builder": config},
        {"knowledge_builder": RecordingProvider()},
        Path(__file__).resolve().parents[1],
    )
    assert builder("experience.summary", {}) == {}
    assert len(invocations) == 2
    assert responses[0].raw["thinking_prefix_incomplete"] is True
    assert (
        responses[0].usage.input_tokens == 2 and responses[0].usage.output_tokens == 2
    )
    assert builder.audit[0]["status"] == "validation_failed"
    assert builder.audit[1]["status"] == "validated"
    assert builder.audit[1]["effective_reasoning_mode"] == "instruct"
