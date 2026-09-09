from types import SimpleNamespace

import pytest

from spatialcraft.models import ContentPart, ModelRequestError, RequestBuilder
from spatialcraft.models.capabilities import ModelCapabilities
from spatialcraft.models.providers.transformers_local import TransformersLocalProvider
from spatialcraft.models.registry import LocalModelConfig, ModelConfig, ProviderKind


def config(path, *, vision):
    return ModelConfig(
        alias="local",
        model_id="local",
        provider=ProviderKind.TRANSFORMERS_LOCAL,
        capabilities=ModelCapabilities(
            supported=frozenset(
                {"text_input", "image_input"} if vision else {"text_input"}
            )
        ),
        local=LocalModelConfig(path=str(path)),
    )


@pytest.mark.parametrize("vision", [True, False])
def test_local_loader_uses_generation_head(monkeypatch, tmp_path, vision):
    import sys

    calls = []
    model = SimpleNamespace(eval=lambda: None)

    def loader(name, result):
        def load(*args, **kwargs):
            calls.append(name)
            return result

        return SimpleNamespace(from_pretrained=load)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=loader("processor", object()),
            AutoModelForImageTextToText=loader("multimodal", model),
            AutoModelForCausalLM=loader("causal", model),
        ),
    )
    monkeypatch.setattr(
        TransformersLocalProvider, "_torch", staticmethod(lambda: object())
    )
    provider = TransformersLocalProvider(config(tmp_path, vision=vision))
    assert provider._load()[0] is model
    assert calls == ["processor", "multimodal" if vision else "causal"]
    provider._load()
    assert len(calls) == 2


def test_template_options_are_explicit_and_managed_keys_rejected(tmp_path):
    cfg = config(tmp_path, vision=False)
    provider = TransformersLocalProvider(cfg)
    seen = {}

    def apply(messages, **kwargs):
        seen.update(kwargs)
        return "prompt"

    processor = SimpleNamespace(apply_chat_template=apply)
    request = (
        RequestBuilder(cfg)
        .user("q")
        .metadata(chat_template_kwargs={"enable_thinking": False})
        .build()
    )
    assert provider._render(processor, request) == ("prompt", [], [])
    assert seen["enable_thinking"] is False
    request = (
        RequestBuilder(cfg)
        .user("q")
        .metadata(chat_template_kwargs={"tokenize": True})
        .build()
    )
    with pytest.raises(ModelRequestError):
        provider._render(processor, request)


def test_media_order_preserves_all_sixteen_images(tmp_path):
    cfg = config(tmp_path, vision=True)
    request = (
        RequestBuilder(cfg)
        .user("q", media=[ContentPart.image_uri(f"/tmp/{i}.png") for i in range(16)])
        .build()
    )
    parts = TransformersLocalProvider._messages(request)[0]["content"]
    assert [part["image"] for part in parts if part["type"] == "image"] == [
        f"/tmp/{i}.png" for i in range(16)
    ]


def test_truncated_native_tool_output_is_not_executed(tmp_path, monkeypatch):
    import torch

    cfg = config(tmp_path, vision=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SimpleNamespace(generate=lambda **kwargs: torch.tensor([[10, 11, 12, 13]]))
    processor = SimpleNamespace(
        batch_decode=lambda *args, **kwargs: [
            "<tool_call><function=detect><parameter=image_uri>unfinished"
        ]
    )
    provider = TransformersLocalProvider(cfg, model=model, processor=processor)
    monkeypatch.setattr(
        provider, "_prepare", lambda *args: {"input_ids": torch.tensor([[10, 11]])}
    )
    response = provider.generate(
        RequestBuilder(cfg).user("question").settings(max_output_tokens=2).build()
    )
    assert response.finish_reason == "length" and not response.tool_calls
    assert response.text.endswith("unfinished")
