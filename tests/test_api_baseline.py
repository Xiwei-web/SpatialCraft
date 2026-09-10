"""Single-response baseline data isolation, scoring, and durable resumption."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from spatialcraft.experiments.protocol import public_task
from spatialcraft.experiments.run_api_baseline import (
    encode_media,
    make_request,
    run_dataset,
    score,
)
from spatialcraft.models.interfaces import ModelRequestError, ModelResponse, TokenUsage
from spatialcraft.models.providers.openai_responses import OpenAIResponsesProvider
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.models.serialization import request_to_dict
from spatialcraft.schemas import AnswerType, ImageInput, TaskSample
from spatialcraft.storage.atomic_io import canonical_json_bytes, sha256_file


@pytest.fixture
def config():
    return ModelConfig.from_dict(
        load_yaml(
            Path(__file__).resolve().parents[1]
            / "configs/models/gpt-5.4-mini-baseline.yaml"
        )
    )


@pytest.fixture
def task(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (20, 20), "red").save(image)
    return TaskSample(
        dataset="sat",
        task_id="task_safe",
        question="Which side?",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        choices=("left", "right"),
        reference_answer="B",
        images=(
            ImageInput(
                uri=str(image),
                media_type="image/png",
                metadata={"secret_mask": "ANNOTATION_SECRET"},
            ),
        ),
        metadata={
            "choice_labels": ["A", "B"],
            "correct_answer_index": 1,
            "reference_answer_text": "ANNOTATION_SECRET",
            "question_type": "direction",
        },
    )


def reply(text="Final Answer: B", **kwargs):
    return ModelResponse(
        provider="openai_responses",
        model="gpt-5.4-mini-2026-03-17",
        text=text,
        finish_reason=kwargs.pop("finish_reason", "completed"),
        usage=TokenUsage(input_tokens=100, output_tokens=5),
        **kwargs,
    )


def test_prompt_only_public_pixels_and_choices(task, config):
    request = make_request(task, config)
    wire = json.dumps(request_to_dict(request))
    assert "ANNOTATION_SECRET" not in wire
    assert "correct_answer_index" not in wire and "reference_answer" not in wire
    assert "A. left" in wire and "B. right" in wire
    assert request.tools == () and request.parallel_tool_calls is False
    assert request.settings.reasoning_effort == "none"
    assert request.settings.max_output_tokens == 4096
    encoded = encode_media(
        request, {task.images[0].uri: sha256_file(task.images[0].uri)}
    )
    payload = OpenAIResponsesProvider(config).build_payload(encoded)
    assert payload["store"] is False and payload["temperature"] == 0
    assert "tools" not in payload and "seed" not in payload
    images = [
        p for m in payload["input"] for p in m["content"] if p["type"] == "input_image"
    ]
    assert images[0]["image_url"].startswith("data:image/png;base64,")
    assert images[0]["detail"] == "high"
    with pytest.raises(ValueError, match="changed"):
        encode_media(request, {task.images[0].uri: "bad"})


@pytest.mark.parametrize(
    "response,expected",
    [
        (reply(), 1.0),
        (reply("Final Answer: A"), 0.0),
        (reply(None), 0.0),
        (
            reply(
                finish_reason="incomplete",
                raw={"incomplete_details": {"reason": "max_output_tokens"}},
            ),
            0.0,
        ),
    ],
)
def test_single_response_scoring(task, response, expected):
    result = score(task, response)
    assert result["reward"] == expected
    assert result["usage"]["output_tokens"] == 5


def test_failed_api_status_is_not_wrong_answer(task):
    with pytest.raises(RuntimeError, match="failed response"):
        score(task, reply(finish_reason="failed"))


def test_existing_numeric_boolean_pointing_rules(task):
    assert (
        score(
            replace(task, answer_type=AnswerType.NUMERIC, reference_answer=2.0),
            reply("Final Answer: 2.01"),
        )["reward"]
        == 1
    )
    assert (
        score(
            replace(task, answer_type=AnswerType.BOOLEAN, reference_answer=True),
            reply("Final Answer: yes"),
        )["reward"]
        == 1
    )
    assert (
        score(
            replace(
                task, answer_type=AnswerType.POINTING, reference_answer=[[0.5, 0.5]]
            ),
            reply("Final Answer: [[0.5, 0.5]]"),
        )["reward"]
        == 1
    )


@pytest.mark.parametrize("requested_model", ["gpt-5.4-mini", "gpt-5.4"])
def test_resume_keeps_api_commits_and_rejects_code_changes(
    tmp_path, task, config, requested_model
):
    config = replace(config, alias=requested_model, model_id=requested_model)
    data = tmp_path / "data/sat"
    data.mkdir(parents=True)
    tasks = (task, replace(task, task_id="task_second"))
    for kind in ("public", "private"):
        (data / f"{kind}.jsonl").write_bytes(
            b"".join(
                canonical_json_bytes(
                    public_task(t) if kind == "public" else t.to_dict()
                )
                for t in tasks
            )
        )
    identity = {
        "count": 2,
        "public_sha256": sha256_file(data / "public.jsonl"),
        "private_sha256": sha256_file(data / "private.jsonl"),
        "media": {task.images[0].uri: sha256_file(task.images[0].uri)},
        "task_ids": [t.task_id for t in tasks],
    }
    manifest = {"datasets": {"sat": identity}}

    class Provider:
        def __init__(self, fail):
            self.calls = 0
            self.fail = fail

        def generate(self, request):
            self.calls += 1
            if self.fail and self.calls == 2:
                raise ModelRequestError("simulated transport failure")
            return replace(reply(), model=requested_model + "-2026-03-05")

    first = Provider(True)
    with pytest.raises(ModelRequestError):
        run_dataset(tmp_path, "sat", config, manifest, "version1", first)
    assert first.calls == 2
    assert not (tmp_path / "sat/results/deployment.json").exists()
    assert json.loads((tmp_path / "sat/progress.json").read_text())["evaluated"] == 1
    commits = {
        p: p.read_bytes() for p in (tmp_path / "sat/stages").rglob("result.json")
    }
    second = Provider(False)
    result = run_dataset(tmp_path, "sat", config, manifest, "version1", second)
    assert second.calls == 1 and result["accuracy"] == 1 and result["evaluated"] == 2
    assert all(p.read_bytes() == old for p, old in commits.items())
    third = Provider(False)
    run_dataset(tmp_path, "sat", config, manifest, "version1", third)
    assert third.calls == 0
    with pytest.raises(ValueError, match="binding changed"):
        run_dataset(tmp_path, "sat", config, manifest, "version2", third)


@pytest.mark.parametrize(
    "returned, valid",
    [
        ("gpt-5.4", True),
        ("gpt-5.4-2026-03-05", True),
        ("gpt-5.4-mini", False),
        ("gpt-5.4-mini-2026-03-17", False),
        ("gpt-5.4-pro", False),
        ("gpt-5.4-nano-2026-03-17", False),
        ("gpt-5.5", False),
    ],
)
def test_model_identity_rejects_sibling_fallback(returned, valid):
    from spatialcraft.experiments.run_api_baseline import validate_response_model

    if valid:
        validate_response_model(returned, "gpt-5.4")
    else:
        with pytest.raises(ValueError, match="Unexpected returned model"):
            validate_response_model(returned, "gpt-5.4")


def test_gpt54_payload_only_changes_model(task, config):
    path = Path(__file__).resolve().parents[1] / "configs/models/gpt-5.4-baseline.yaml"
    large = ModelConfig.from_dict(load_yaml(path))
    mini_payload = OpenAIResponsesProvider(config).build_payload(
        make_request(task, config)
    )
    large_payload = OpenAIResponsesProvider(large).build_payload(
        make_request(task, large)
    )
    assert large_payload == {**mini_payload, "model": "gpt-5.4"}
    assert large.generation.seed is None and large.generation.reasoning_effort == "none"
