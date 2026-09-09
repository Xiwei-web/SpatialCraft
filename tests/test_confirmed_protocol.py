from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.knowledge.experience import (
    ExperienceBank,
    ExperienceIndex,
    HashingEmbedder,
)
from spatialcraft.knowledge.experience.contextual_rewriter import (
    ContextualExperienceRewriter,
)
from spatialcraft.knowledge.experience.maintenance import ExperienceMaintenance
from spatialcraft.knowledge.experience.retriever import ExperienceRetriever
from spatialcraft.knowledge.skill import (
    SeedCatalog,
    SkillMaintenance,
    SkillPool,
    SkillSelector,
)
from spatialcraft.models import (
    APIConfig,
    ContentPart,
    MessageRole,
    ModelCapabilities,
    ModelConfig,
    ModelConfigurationError,
    ModelMessage,
    ModelRequestError,
    ProviderKind,
    RequestBuilder,
)
from spatialcraft.models.providers.openai_embeddings import OpenAIEmbeddingsProvider
from spatialcraft.models.providers.transformers_local import TransformersLocalProvider
from spatialcraft.models.response_parser import parse_generated_text
from spatialcraft.models.scoring import with_skill_prompt
from spatialcraft.models.serialization import request_from_dict, request_to_dict
from spatialcraft.schemas import (
    ExperienceItem,
    ImageInput,
    SkillItem,
    SkillStats,
    TaskSample,
)
from spatialcraft.storage.atomic_io import atomic_write_json, read_json
from spatialcraft.verification.exact_match import normalize_answer


def embedding_config():
    return ModelConfig(
        alias="text-embedding-3-small",
        model_id="text-embedding-3-small",
        provider=ProviderKind.OPENAI_EMBEDDINGS,
        api=APIConfig(api_key_env="OPENAI_API_KEY"),
        capabilities=ModelCapabilities(
            supported=frozenset({"text_input", "embeddings"})
        ),
        metadata={"dimensions": 2},
    )


class FakeEmbeddings:
    def __init__(self):
        self.calls = []
        self.embeddings = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            usage=SimpleNamespace(prompt_tokens=7),
            data=[
                SimpleNamespace(index=i, embedding=[3.0, 4.0])
                for i in reversed(range(len(kwargs["input"])))
            ],
        )


def test_embedding_lazy_cache_order_empty_and_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = FakeEmbeddings()
    provider = OpenAIEmbeddingsProvider(
        embedding_config(), client=client, cache_dir=tmp_path, token_counter=len
    )
    assert provider.embed(()) == () and not client.calls
    values = provider.embed(("alpha", "beta", "alpha"))
    assert values == ((0.6, 0.8),) * 3
    assert len(client.calls) == 1 and client.calls[0]["input"] == ["alpha", "beta"]
    assert client.calls[0]["encoding_format"] == "float"
    usage = list((tmp_path / "usage").glob("*.json"))
    assert len(usage) == 1 and read_json(usage[0])["input_tokens"] == 7
    assert "alpha" not in usage[0].read_text()
    resumed = OpenAIEmbeddingsProvider(
        embedding_config(), cache_dir=tmp_path, token_counter=len
    )
    assert resumed.embed(("beta", "alpha")) == ((0.6, 0.8),) * 2
    assert resumed._client is None
    assert len(list((tmp_path / "usage").glob("*.json"))) == 1
    with pytest.raises(ModelConfigurationError):
        resumed.embed(("uncached",))
    with pytest.raises(ValueError):
        provider.embed(("",))
    with pytest.raises(ModelRequestError):
        provider.embed(("x" * 8193,))


def test_embedding_cache_tamper_rejected(tmp_path):
    provider = OpenAIEmbeddingsProvider(
        embedding_config(),
        client=FakeEmbeddings(),
        cache_dir=tmp_path,
        token_counter=len,
    )
    provider.embed(("alpha",))
    path = next(tmp_path.glob("*.json"))
    item = read_json(path)
    item["vector"][0] = 0.7
    atomic_write_json(path, item)
    with pytest.raises(ModelRequestError):
        OpenAIEmbeddingsProvider(
            embedding_config(), cache_dir=tmp_path, token_counter=len
        ).embed(("alpha",))


@pytest.mark.parametrize("bad", ["indices", "model", "nonfinite", "dimensions"])
def test_embedding_response_validation(bad):
    class Broken(FakeEmbeddings):
        def create(self, **kwargs):
            result = super().create(**kwargs)
            if bad == "indices":
                result.data[0].index = 4
            elif bad == "model":
                result.model = "other"
            elif bad == "nonfinite":
                result.data[0].embedding = [float("nan"), 1]
            else:
                result.data[0].embedding = [1]
            return result

    with pytest.raises(ModelRequestError):
        OpenAIEmbeddingsProvider(
            embedding_config(), client=Broken(), token_counter=len
        ).embed(("a",))


def test_top3_per_subtask_union_and_rewrite_filter():
    bank = ExperienceBank(
        tuple(
            ExperienceItem(
                experience_id=f"e{i}", condition=f"condition{i}", action="act"
            )
            for i in range(6)
        )
    )

    class Index:
        def search(self, query, embedder, top_k):
            assert top_k == 3
            start = 0 if query == "subtask1" else 3
            return tuple((f"e{i}@1", 0.9) for i in range(start, start + 3))

    decomposer = SimpleNamespace(decompose=lambda task: ("subtask1", "subtask2"))
    task = TaskSample(dataset="test", question="question", reference_answer="secret")
    retrieved = ExperienceRetriever(
        bank, Index(), HashingEmbedder(), decomposer=decomposer
    ).retrieve(task)
    assert len(retrieved) == 6  # no global top-5 truncation

    def rewrite(prompt):
        assert "secret" not in prompt
        return "SKIP" if "condition0" in prompt else "use when observed"

    result = ContextualExperienceRewriter(rewrite).rewrite(task, retrieved)
    assert len(result) == 5 and all(
        r.prompt_text == "use when observed" for r in result
    )


def test_index_rejects_different_space():
    bank = ExperienceBank((ExperienceItem(condition="left", action="compare"),))
    index = ExperienceIndex.build(bank, HashingEmbedder(32)).freeze()
    with pytest.raises(ValueError, match="space"):
        index.search("left", HashingEmbedder(64))


def test_capacities_and_archived_history_preserved():
    bank = ExperienceBank(
        tuple(
            ExperienceItem(condition=f"condition{i}", action=f"action{i}")
            for i in range(105)
        )
    )
    limited, operations = ExperienceMaintenance().enforce_capacity(bank)
    assert len(limited.active()) == 100 and len(limited.all()) == 105 and operations
    pool = SkillPool(
        tuple(
            SkillItem(
                skill_id=f"s{i}",
                name=f"s{i}",
                initiation=f"when{i}",
                policy=(f"do{i}",),
                termination=f"done{i}",
                stats=SkillStats(frequency=1, total_gain=i, average_gain=i),
            )
            for i in range(24)
        )
    )
    pruned, refs = SkillMaintenance().enforce_capacity(pool)
    assert len(pruned.active()) == 20 and len(pruned.all()) == 24 and len(refs) == 4
    assert {s.skill_id for s in pruned.active()} == {f"s{i}" for i in range(4, 24)}
    with pytest.raises(ValueError):
        SkillMaintenance().enforce_capacity(pruned.freeze())


def test_exactly_one_embedding_argmax_and_safe_task():
    seen = []

    class Embedder:
        def embed(self, texts):
            seen.append(texts)
            return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    pool = SkillPool(
        (
            SkillItem(
                skill_id="a",
                name="a",
                initiation="start-a",
                policy=("pa",),
                termination="ta",
            ),
            SkillItem(
                skill_id="b",
                name="b",
                initiation="start-b",
                policy=("pb",),
                termination="tb",
            ),
        )
    )
    from spatialcraft.agent import StateBuilder

    task = TaskSample(
        dataset="test",
        question="left?",
        reference_answer="secret",
        metadata={"mask_uri": "secret-mask", "question_type": "relation"},
        images=(ImageInput(uri="image.png", metadata={"gt": "secret"}),),
    )
    assert (
        SkillSelector(embedder=Embedder())
        .select(pool, task, StateBuilder().initial(task))
        .skill_id
        == "b"
    )
    assert seen[0][1:] == ["start-a", "start-b"]
    safe = task.without_reference_answer().to_dict()
    assert "secret" not in str(safe)


def test_native_tools_and_final_answer_extraction():
    response = parse_generated_text(
        '<think>internal</think><tool_call>\n<function=detect>\n<parameter=queries>["cup"]</parameter>\n<parameter=threshold>0.2</parameter>\n</function>\n</tool_call>',
        provider="local",
        model="qwen",
    )
    assert response.tool_calls[0].arguments == {"queries": ["cup"], "threshold": 0.2}
    from spatialcraft.models import ModelResponseError

    with pytest.raises(ModelResponseError):
        parse_generated_text(
            "<tool_call><function=detect>", provider="local", model="qwen"
        )
    assert normalize_answer("The evidence is clear.\nFinal Answer: Yes") == "yes"


def test_exact_skill_slot_codec_preserves_images(tmp_path):
    from spatialcraft.models.registry import LocalModelConfig

    cfg = ModelConfig(
        alias="qwen",
        model_id="qwen",
        provider=ProviderKind.TRANSFORMERS_LOCAL,
        capabilities=ModelCapabilities(
            supported=frozenset({"text_input", "image_input"})
        ),
        local=LocalModelConfig(path=str(tmp_path)),
    )
    parent = SeedCatalog.pool().active()[0]
    old = (
        RequestBuilder(cfg)
        .system("priority")
        .message(
            ModelMessage.text(
                MessageRole.DEVELOPER,
                "Active procedural skill:\n" + parent.format_for_prompt(),
                metadata={"ppo_skill_slot": True},
            )
        )
        .user("q", media=(ContentPart.image_bytes(b"bytes"),))
        .build()
    )
    restored = request_from_dict(request_to_dict(old))
    child = replace(parent, name="replacement", policy=("new-policy",))
    new = with_skill_prompt(restored, child)
    assert sum(bool(m.metadata.get("ppo_skill_slot")) for m in new.messages) == 1
    assert "new-policy" in "\n".join(m.text_content for m in new.messages)
    assert new.messages[-1].content[-1].data == b"bytes"
    native = TransformersLocalProvider._messages(new)
    assert [m["role"] for m in native] == ["system", "user"]


@pytest.mark.parametrize(
    "values",
    [
        {"rollouts_per_task": 1},
        {"accumulation_passes": 10},
        {"skill_candidates": 1},
        {"experience_capacity": 101},
        {"skill_capacity": 21},
        {"training_temperature": 0},
        {"embedding_model": "qwen"},
    ],
)
def test_settings_enforce_confirmed_protocol(values):
    with pytest.raises(ValueError):
        ExperimentSettings(**values)
