"""Offline behavioral checks; deliberately not executed during implementation."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from RAG.core import (
    ExampleIndex,
    embedding_text,
    make_record,
    make_request,
    representatives,
)
from RAG.data import validate_splits
from RAG.runner import load_memory, run_dataset
from spatialcraft.experiments.protocol import public_task
from spatialcraft.models import ModelResponse, TokenUsage
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.models.serialization import request_to_dict
from spatialcraft.schemas import AnswerType, ImageInput, TaskSample
from spatialcraft.storage.atomic_io import canonical_json_bytes, sha256_file

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def config():
    return ModelConfig.from_dict(
        load_yaml(PROJECT / "configs/models/gpt-5.4-mini-baseline.yaml")
    )


def task(identifier, *, dataset="sat"):
    return TaskSample(
        dataset=dataset,
        task_id=identifier,
        question=f"Which side in {identifier}?",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        choices=("left", "right"),
        reference_answer="PRIVATE_LABEL_SENTINEL",
        metadata={
            "question_type": "direction",
            "choice_labels": ["A", "B"],
            "private_annotation": "PRIVATE_METADATA_SENTINEL",
        },
    )


def response(text="Final Answer: A", *, finish="completed"):
    return ModelResponse(
        provider="openai_responses",
        model="gpt-5.4-mini",
        text=text,
        finish_reason=finish,
        usage=TokenUsage(input_tokens=20, output_tokens=4),
    )


def test_private_labels_never_enter_embedding_records_or_prompts(config):
    previous, current = task("previous"), task("current")
    record = make_record(previous, response("RAW PRIOR OUTPUT"), 0)
    request = make_request(
        current,
        config,
        stage="deployment",
        examples=({"record": record, "cosine": 1.0},),
    )
    wire = json.dumps([record, request_to_dict(request), embedding_text(current)])
    assert "PRIVATE_LABEL_SENTINEL" not in wire
    assert "PRIVATE_METADATA_SENTINEL" not in wire
    assert "RAW PRIOR OUTPUT" in wire and "RAW PRIOR OUTPUT" not in embedding_text(
        previous
    )
    assert record["model_output"] == "RAW PRIOR OUTPUT"
    assert "reward" not in record and "summary" not in record and "lesson" not in record
    assert request.tools == () and request.settings.reasoning_effort == "none"


def test_representatives_do_not_filter_incorrect_answers():
    previous = replace(task("previous"), reference_answer="B")
    wrong = make_record(previous, response("Final Answer: A"), 0)
    correct = make_record(previous, response("Final Answer: B"), 1)
    assert representatives([correct, wrong]) == (wrong,)
    truncated = make_record(task("truncated"), response(finish="incomplete"), 0)
    assert representatives([wrong, truncated]) == (wrong,)


def test_cosine_retrieval_is_distinct_same_dataset_and_not_self():
    records = [
        make_record(task(name, dataset=dataset), response(), rollout)
        for name, dataset, rollout in [
            ("a", "sat", 0),
            ("a", "sat", 1),
            ("b", "sat", 0),
            ("c", "erqa", 0),
        ]
    ]
    chosen = representatives(records)
    index = ExampleIndex(
        records,
        {
            "record_ids": [r["record_id"] for r in chosen],
            "dimensions": 2,
            "vectors": [[10, 0], [1, 1], [20, 0]],
        },
    )
    hits = index.retrieve(task("query"), [1, 0], 3)
    assert [r["record"]["task_id"] for r in hits] == ["a", "b"]
    assert hits[0]["cosine"] == pytest.approx(1.0)
    assert [r["record"]["task_id"] for r in index.retrieve(task("a"), [1, 0], 3)] == [
        "b"
    ]
    with pytest.raises(ValueError):
        index.retrieve(task("query"), [float("nan"), 0], 3)


def test_environment_cannot_retrieve(config):
    previous = make_record(task("previous"), response(), 0)
    with pytest.raises(ValueError, match="must not retrieve"):
        make_request(
            task("current"),
            config,
            stage="environment",
            examples=({"record": previous},),
        )


def test_exact_task_overlap_rejected_but_shared_image_audited():
    image = ImageInput(uri="/example.png")
    first = TaskSample.from_dict(public_task(replace(task("a"), images=(image,))))
    duplicate = replace(first, task_id="different-source-id")
    media = {image.uri: "image-hash"}
    with pytest.raises(ValueError, match="Exact"):
        validate_splits("sat", [first], [duplicate], media)
    distinct = replace(duplicate, question="A different question on the same image")
    assert validate_splits("sat", [first], [distinct], media) == 1


class FakeEmbeddings:
    identity = "offline-fake-embedding:2"
    dimensions = 2

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        assert all("PRIVATE_LABEL_SENTINEL" not in text for text in texts)
        return tuple((1.0, 0.0) for _ in texts)


class FakeProvider:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return response("Final Answer: B")


def test_environment_deployment_resume_and_frozen_corpus(tmp_path, config):
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(image)
    previous = replace(
        task("prior"), images=(ImageInput(uri=str(image), media_type="image/png"),)
    )
    current = replace(task("current"), reference_answer="B", images=previous.images)
    data = tmp_path / "data/sat"
    data.mkdir(parents=True)
    for filename, rows in (
        ("environment", [public_task(previous)]),
        ("public", [public_task(current)]),
        ("private", [current.to_dict()]),
    ):
        (data / f"{filename}.jsonl").write_bytes(
            b"".join(canonical_json_bytes(r) for r in rows)
        )
    embeddings, provider = FakeEmbeddings(), FakeProvider()
    identity = {"media": {str(image): sha256_file(image)}}
    options = {"environment_rollouts": 2, "top_k": 3}
    args = (
        tmp_path,
        "sat",
        identity,
        {"policy": "offline-test"},
        config,
        config,
        options,
        embeddings,
    )
    run_dataset(*args, "environment", provider=provider)
    memory = tmp_path / "sat/memory"
    before = {p.name: p.read_bytes() for p in memory.iterdir()}
    report = run_dataset(*args, "deployment", provider=provider)
    assert report["deployment"]["accuracy"] == 1
    assert len(provider.requests) == 3
    assert [r.metadata["retrieved_count"] for r in provider.requests] == [0, 0, 1]
    assert {p.name: p.read_bytes() for p in memory.iterdir()} == before
    calls = embeddings.calls
    run_dataset(*args, "all", provider=provider)
    assert len(provider.requests) == 3 and embeddings.calls == calls
    snapshot = json.loads((memory / "snapshot.json").read_text())
    (memory / "records.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        load_memory(tmp_path / "sat", snapshot["binding_sha256"], embeddings.identity)


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "gpt-5.4"])
@pytest.mark.parametrize("effort", ["none", "medium"])
@pytest.mark.parametrize("max_tokens", [4096, 16384])
def test_reasoning_effort_and_wire_sampling_parameters(
    model, effort, max_tokens, tmp_path
):
    from RAG.run_rag import configuration, parser
    from spatialcraft.models.providers.openai_responses import OpenAIResponsesProvider

    args = parser().parse_args(
        [
            "--model",
            model,
            "--output",
            str(tmp_path),
            "--reasoning-effort",
            effort,
            "--max-output-tokens",
            str(max_tokens),
        ]
    )
    environment, deployment, _ = configuration(args)
    for stage, config in (("environment", environment), ("deployment", deployment)):
        request = make_request(task("query"), config, stage=stage)
        payload = OpenAIResponsesProvider(config).build_payload(request)
        assert payload["reasoning"] == {"effort": effort}
        assert payload["max_output_tokens"] == max_tokens
        assert "top_p" not in payload and "seed" not in payload
        if effort == "medium":
            assert "temperature" not in payload
            assert request.settings.temperature is None
        else:
            assert payload["temperature"] == (0.7 if stage == "environment" else 0.0)


def test_retrieval_excludes_exact_content_with_different_ids_and_image_paths():
    first = replace(task("prior"), images=(ImageInput(uri="/prior.png"),))
    duplicate = replace(
        first, task_id="deployment", images=(ImageInput(uri="/alias.png"),)
    )
    other = replace(first, task_id="other", question="A different spatial question")
    records = [make_record(first, response(), 0), make_record(other, response(), 0)]
    selected = representatives(records)
    index = ExampleIndex(
        records,
        {
            "record_ids": [row["record_id"] for row in selected],
            "dimensions": 2,
            "vectors": [[1, 0], [1, 0]],
        },
    )
    media = {"/prior.png": "same-image-bytes", "/alias.png": "same-image-bytes"}
    hits = index.retrieve(duplicate, [1, 0], 3, media=media)
    assert [hit["record"]["task_id"] for hit in hits] == ["other"]
    assert len(index.exact_matches(duplicate, media)) == 1
