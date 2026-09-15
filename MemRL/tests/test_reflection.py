"""The R/GT boundary and durable visual-reflection calls, without external APIs."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from memp.runner import Settings, model_config
from MemRL.reflection import ReflectionBuilder, _ReflectionResponsesProvider
from spatialcraft.experiments.journal import RunJournal
from spatialcraft.experiments.protocol import public_task
from spatialcraft.experiments.react_api import TaskMedia
from spatialcraft.models import ContentKind, ModelResponse, ResponseToolCall, TokenUsage
from spatialcraft.models.serialization import request_to_dict
from spatialcraft.schemas import AnswerType, ImageInput, TaskSample, TaskSplit
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import sha256_file

PROJECT = Path(__file__).resolve().parents[2]
SECRET = "PRIVATE_REFERENCE_ANNOTATION"
ANSWER = "[(0.21, 0.37), (0.22, 0.38)]"
REFLECTION = "Identify the observer frame and verify the normalized point against visible free space."


class FakeProvider:
    def __init__(self, *, invalid=None):
        self.requests, self.invalid = [], invalid
        self.close_count = 0
        self._client = SimpleNamespace(close=self.close)

    def close(self):
        self.close_count += 1

    def factory(self, config, media):
        self.config, self.media = config, media
        return self

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(
            provider="fixture",
            model="gpt-5.4" if self.invalid == "model" else request.model_alias,
            text="" if self.invalid == "empty" else REFLECTION,
            finish_reason="incomplete" if self.invalid == "length" else "completed",
            tool_calls=(ResponseToolCall(name="Invented", arguments={}),)
            if self.invalid == "tool"
            else (),
            raw={"output": [{"type": "function_call", "arguments": "{"}]}
            if self.invalid == "raw_tool"
            else {},
            usage=TokenUsage(input_tokens=20, output_tokens=17, reasoning_tokens=9),
        )


@pytest.fixture
def case(tmp_path):
    images = []
    for index in range(2):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (8, 8), (80 * index, 10, 20)).save(path)
        images.append(
            ImageInput(
                uri=str(path), media_type="image/png", metadata={"private": SECRET}
            )
        )
    private = TaskSample(
        dataset="robospatial",
        task_id="environment-1",
        question="Point in the open space.",
        split=TaskSplit.TRAIN,
        answer_type=AnswerType.POINTING,
        images=tuple(images),
        reference_answer=ANSWER,
        metadata={
            "experiment_split": "environment",
            "question_type": "context",
            "mask_uri": SECRET,
            "depth_uri": SECRET,
        },
    )
    task = TaskSample.from_dict(public_task(private))
    media = TaskMedia(
        task,
        {"media": {image.uri: sha256_file(image.uri) for image in images}},
        StorageLayout(tmp_path / "artifacts"),
    )
    config = model_config(
        PROJECT,
        Settings(
            model="gpt-5.4-mini", reasoning_effort="medium", max_output_tokens=16384
        ),
    )
    provider = FakeProvider()
    return SimpleNamespace(
        config=config,
        private=private,
        task=task,
        media=media,
        provider=provider,
        journal=RunJournal(tmp_path / "journal", {"fixture": True}),
        kwargs={
            "task": task,
            "reference": None,
            "trace": [{"step": 0, "final_answer": "Final Answer: [[0.3, 0.5]]"}],
            "model_output": {
                "text": "Final Answer: [[0.3, 0.5]]",
                "accepted_final": "[[0.3, 0.5]]",
                "tool_calls": [],
            },
            "reward": 0.0,
            "media": media,
        },
    )


def build(case, *, variant="R", **changes):
    builder = ReflectionBuilder(case.config, variant, case.provider.factory)
    kwargs = {**case.kwargs, **changes}
    if variant == "GT" and "reference" not in changes:
        kwargs["reference"] = case.private
    value = builder.build(case.journal, variant, **kwargs)
    return value, builder


def test_r_and_gt_share_visual_context_and_only_gt_gets_answer(case):
    assert build(case)[0] == REFLECTION
    assert build(case, variant="GT")[0] == REFLECTION
    r_request, gt_request = case.provider.requests
    r_json = json.loads(r_request.messages[-1].content[0].text)
    gt_json = json.loads(gt_request.messages[-1].content[0].text)
    assert gt_json.pop("correct_answer") == ANSWER
    assert r_json == gt_json
    assert r_json["scalar_reward"] == 0.0
    assert r_request.messages[0] == gt_request.messages[0]
    for request in case.provider.requests:
        assert not request.tools and not request.parallel_tool_calls
        assert request.settings.max_output_tokens == 16384
        assert request.settings.reasoning_effort == "medium"
        assert request.settings.temperature is None and request.settings.seed is None
        assert SECRET not in json.dumps(request_to_dict(request))
        images = [
            part
            for part in request.messages[-1].content
            if part.kind is ContentKind.IMAGE
        ]
        assert [part.uri for part in images] == [
            image.uri for image in case.task.images
        ]
        assert [part.detail for part in images] == ["high", "high"]
        assert request.metadata["operation"] == "memrl.reflection"
        assert request.metadata["phase"] == "environment"
    assert ANSWER not in json.dumps(request_to_dict(r_request))


@pytest.mark.parametrize("reward", [0, 1, 0.0, 1.0])
def test_success_and_failure_both_produce_reflections(case, reward):
    assert build(case, reward=reward)[0] == REFLECTION


@pytest.mark.parametrize(
    "reward", [None, True, -1, 2, 0.5, float("nan"), float("inf"), "1"]
)
def test_invalid_reward_rejected_before_provider(case, reward):
    with pytest.raises(ValueError, match="binary scalar"):
        build(case, reward=reward)
    assert not case.provider.requests


@pytest.mark.parametrize("variant", ["R", "GT"])
@pytest.mark.parametrize("change", ["phase", "split", "metadata"])
def test_reflections_cannot_run_on_heldout_tasks(case, variant, change):
    changes = (
        {"phase": "deployment"}
        if change == "phase"
        else {
            "task": replace(case.task, split=TaskSplit.TEST)
            if change == "split"
            else replace(
                case.task,
                metadata={**case.task.metadata, "experiment_split": "deployment"},
            )
        }
    )
    with pytest.raises(ValueError, match="environment"):
        build(case, variant=variant, **changes)
    assert not case.provider.requests


def test_r_rejects_reference_even_if_not_used(case):
    with pytest.raises(ValueError, match="must not receive"):
        build(case, reference=case.private)
    assert not case.provider.requests


@pytest.mark.parametrize("reference", [None, "missing", "mismatch"])
def test_gt_requires_complete_matching_reference(case, reference):
    private = (
        None
        if reference is None
        else replace(
            case.private,
            **(
                {"reference_answer": None}
                if reference == "missing"
                else {"question": "Different question"}
            ),
        )
    )
    with pytest.raises(ValueError, match="reference"):
        build(case, variant="GT", reference=private)
    assert not case.provider.requests


def test_private_task_rejected_instead_of_forwarded(case):
    with pytest.raises(ValueError, match="sanitized"):
        build(case, task=case.private)
    assert not case.provider.requests


@pytest.mark.parametrize(
    "changes",
    [
        {
            "model_output": {
                "text": "candidate",
                "verifier": {"expected_answer": SECRET},
            }
        },
        {"trace": [{"observations": [{"verifier_details": {"answer": SECRET}}]}]},
        {
            "model_output": {
                "text": "candidate",
                "tool_calls": [{"reference_answer": SECRET}],
            }
        },
    ],
)
def test_verifier_details_cannot_enter_visible_evidence(case, changes):
    with pytest.raises(ValueError, match="fields"):
        build(case, **changes)
    assert not case.provider.requests


@pytest.mark.parametrize("invalid", ["empty", "length", "model", "tool", "raw_tool"])
def test_invalid_paid_response_is_committed_and_not_regenerated(case, invalid):
    case.provider.invalid = invalid
    for _ in range(2):
        with pytest.raises(ValueError):
            build(case)
    assert len(case.provider.requests) == 1
    assert case.journal.read_committed("R")["usage"]["reasoning_tokens"] == 9


def test_valid_response_replays_and_external_client_is_not_closed(case):
    for _ in range(2):
        value, builder = build(case)
        assert value == REFLECTION
        builder.close()
    assert len(case.provider.requests) == 1
    assert case.provider.close_count == 0


def test_changed_media_is_rejected_before_paid_call_or_replay(case):
    build(case)
    Path(case.task.images[0].uri).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="content changed"):
        build(case)
    assert len(case.provider.requests) == 1


@pytest.mark.parametrize("invalid", ["model", "status", "malformed_call"])
def test_real_provider_path_commits_invalid_raw_output_before_validation(
    case, invalid, monkeypatch
):
    calls = []
    raw = {
        "id": "fixture",
        "model": "gpt-5.4" if invalid == "model" else "gpt-5.4-mini",
        "status": "failed" if invalid == "status" else "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": REFLECTION}],
            }
        ],
        "usage": {
            "input_tokens": 30,
            "output_tokens": 12,
            "output_tokens_details": {"reasoning_tokens": 4},
        },
    }
    if invalid == "malformed_call":
        raw["output"].append(
            {"type": "function_call", "name": "Oops", "arguments": "{"}
        )
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **payload: calls.append(payload) or raw)
    )
    monkeypatch.setattr(
        _ReflectionResponsesProvider, "_make_client", lambda self: client
    )
    builder = ReflectionBuilder(case.config)
    for _ in range(2):
        with pytest.raises(ValueError):
            builder.build(case.journal, "real", **case.kwargs)
    assert len(calls) == 1
    committed = case.journal.read_committed("real")
    assert committed["raw"] == raw
    assert committed["usage"]["input_tokens"] == 30
    images = [
        part
        for entry in calls[0]["input"]
        for part in entry.get("content", ())
        if part["type"] == "input_image"
    ]
    assert len(images) == 2 and all(part["detail"] == "high" for part in images)
    assert all(
        part["image_url"].startswith("data:image/png;base64,") for part in images
    )


def test_close_only_owned_client_once(case, monkeypatch):
    owned = FakeProvider()
    monkeypatch.setattr(
        "MemRL.reflection._ReflectionResponsesProvider", lambda config, media: owned
    )
    builder = ReflectionBuilder(case.config)
    builder.build(case.journal, "owned", **case.kwargs)
    builder.close()
    builder.close()
    assert owned.close_count == 1


def test_unknown_variant_is_rejected(case):
    with pytest.raises(ValueError, match="R or GT"):
        ReflectionBuilder(case.config, "OTHER")


@pytest.mark.parametrize("variant", ["R", "GT"])
def test_public_tool_mask_and_depth_are_valid_evidence_without_private_metadata(
    case, variant
):
    trace = [
        {
            "step": 1,
            "tool_calls": [
                {
                    "name": "Pose",
                    "arguments": {
                        "mask_uri": "MEMORY_ARTIFACT_2",
                        "depth_uri": "MEMORY_ARTIFACT_3",
                    },
                }
            ],
            "observations": [
                {
                    "structured_output": {
                        "mask_uri": "MEMORY_ARTIFACT_2",
                        "depth_uri": "MEMORY_ARTIFACT_3",
                    }
                }
            ],
        }
    ]
    assert build(case, variant=variant, trace=trace)[0] == REFLECTION
    request = case.provider.requests[0]
    payload = json.loads(request.messages[-1].content[0].text)
    assert payload["trajectory"] == trace
    assert SECRET not in json.dumps(request_to_dict(request))
