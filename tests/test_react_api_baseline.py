"""Native API/tool round trips, scoped evidence, recovery, and durable resumption."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from spatialcraft.agent import StateBuilder
from spatialcraft.experiments.protocol import public_task
from spatialcraft.experiments.react_api import (
    ReActResponsesProvider,
    TaskMedia,
    ToolOnlyComposer,
    parse_react_response,
)
from spatialcraft.experiments.run_react_baseline import run_dataset
from spatialcraft.models import ModelRequestError
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.models.serialization import request_to_dict
from spatialcraft.schemas import AnswerType, ImageInput, TaskSample
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import canonical_json_bytes, sha256_file
from spatialcraft.tools.real import create_real_tool_registry


@pytest.fixture
def config():
    return ModelConfig.from_dict(
        load_yaml(
            Path(__file__).resolve().parents[1]
            / "configs/models/gpt-5.4-mini-react.yaml"
        )
    )


@pytest.fixture
def prepared(tmp_path):
    image = tmp_path / "red.png"
    Image.new("RGB", (20, 20), "red").save(image)
    task = TaskSample(
        dataset="sat",
        task_id="first",
        question="Which side?",
        choices=("left", "right"),
        answer_type=AnswerType.MULTIPLE_CHOICE,
        reference_answer="B",
        images=(
            ImageInput(
                uri=str(image),
                media_type="image/png",
                metadata={"secret": "SECRET_LABEL"},
            ),
        ),
        metadata={
            "choice_labels": ["A", "B"],
            "correct_answer_index": 1,
            "reference_answer_text": "SECRET_LABEL",
            "question_type": "direction",
        },
    )
    data = tmp_path / "data/sat"
    data.mkdir(parents=True)
    for kind in ("public", "private"):
        (data / f"{kind}.jsonl").write_bytes(
            canonical_json_bytes(
                public_task(task) if kind == "public" else task.to_dict()
            )
        )
    identity = {
        "count": 1,
        "task_ids": ["first"],
        "media": {str(image): sha256_file(image)},
        **{
            f"{kind}_sha256": sha256_file(data / f"{kind}.jsonl")
            for kind in ("public", "private")
        },
    }
    return task, {"datasets": {"sat": identity}}


def raw(
    *,
    tool=None,
    arguments=None,
    text=None,
    truncated=False,
    model="gpt-5.4-mini-2026-03-17",
):
    output = []
    if tool:
        output.append(
            {
                "type": "function_call",
                "name": tool,
                "call_id": "call_" + tool,
                "arguments": arguments
                if isinstance(arguments, str)
                else json.dumps(arguments),
            }
        )
    if text is not None:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )
    return {
        "id": "response-test",
        "status": "incomplete" if truncated else "completed",
        "model": model,
        "output": output,
        "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
        "usage": {
            "input_tokens": 20,
            "output_tokens": 4096 if truncated else 8,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def geometry(**kwargs):
    return raw(
        tool="geometry",
        arguments={"operation": "point_distance", "first": [0, 0], "second": [3, 4]},
        **kwargs,
    )


class Client:
    def __init__(self, responses):
        self.queue = iter(responses)
        self.calls = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **payload):
        self.calls.append(payload)
        response = next(self.queue)
        if isinstance(response, Exception):
            raise response
        return response

    def factory(self, config, media):
        return ReActResponsesProvider(config, media, client=self)


def execute(tmp_path, config, prepared, client, max_steps=50):
    return run_dataset(
        tmp_path,
        "sat",
        config,
        prepared[1],
        {"code_sha256": "v1"},
        max_steps=max_steps,
        provider_factory=client.factory,
    )


def test_prompt_and_media_are_public_without_memory(tmp_path, config, prepared):
    task, manifest = prepared
    media = TaskMedia(
        task, manifest["datasets"]["sat"], StorageLayout(tmp_path / "store")
    )
    composer = ToolOnlyComposer(config, create_real_tool_registry(), media=media)
    state = StateBuilder().initial(TaskSample.from_dict(public_task(task)))
    request = composer.compose(task, state)
    wire = json.dumps(request_to_dict(request))
    assert "SECRET_LABEL" not in wire and "correct_answer_index" not in wire
    assert "Active procedural skill" not in wire and "Retrieved experience" not in wire
    assert "A. left" in wire and "B. right" in wire
    assert len(request.tools) == 11 and all(t.strict is False for t in request.tools)
    assert not request.parallel_tool_calls and request.settings.seed is None
    assert (
        request.settings.reasoning_effort == "none"
        and request.settings.max_output_tokens == 4096
    )
    with pytest.raises(ValueError, match="cannot contain"):
        composer.compose(
            task, replace(state, metadata={"active_skill_prompt": "SECRET_SKILL"})
        )
    with pytest.raises(ValueError, match="not an input"):
        media.check_uri(str(tmp_path / "data/sat/private.jsonl"))


def test_native_round_trip_with_real_geometry_and_draw(tmp_path, config, prepared):
    client = Client(
        [
            geometry(),
            raw(
                tool="draw",
                arguments={
                    "image_uri": prepared[0].images[0].uri,
                    "points": [{"point": [4, 4]}],
                },
            ),
            raw(text="Final Answer: B"),
        ]
    )
    report = execute(tmp_path, config, prepared, client)
    assert report["accuracy"] == 1 and report["tool_usage"] == {
        "geometry": 1,
        "draw": 1,
    }
    assert report["call_counts"]["normal_llm_calls"] == 3
    assert report["usage"]["output_tokens"] == 24
    assert all("seed" not in call for call in client.calls)
    second = client.calls[1]
    assistant_history = [m for m in second["input"] if m.get("role") == "assistant"]
    assert assistant_history and all(
        p["type"] == "output_text" for m in assistant_history for p in m["content"]
    )
    outputs = (
        [i for i in second["input"] if i["type"] == "function_call_output"]
        if all("type" in i for i in second["input"])
        else [i for i in second["input"] if i.get("type") == "function_call_output"]
    )
    assert outputs[0]["call_id"] == "call_geometry"
    assert json.loads(outputs[0]["output"])["structured_output"]["distance"] == 5
    assert "json_artifact_observations" in outputs[0]["output"]
    images = [
        part
        for item in client.calls[2]["input"]
        for part in item.get("content", [])
        if part["type"] == "input_image"
    ]
    assert len(images) == 2 and all(p["detail"] == "high" for p in images)
    assert all(p["image_url"].startswith("data:image/") for p in images)
    commits = {
        p: p.read_bytes() for p in (tmp_path / "sat/stages").rglob("result.json")
    }
    replay = Client([])
    repeated = execute(tmp_path, config, prepared, replay)
    assert repeated["accuracy"] == 1 and not replay.calls
    assert all(p.read_bytes() == content for p, content in commits.items())


def test_partial_native_arguments_recover_same_step(tmp_path, config, prepared):
    client = Client(
        [
            raw(tool="geometry", arguments='{"operation":', truncated=True),
            geometry(),
            raw(text="Final Answer: B"),
        ]
    )
    report = execute(tmp_path, config, prepared, client)
    assert report["accuracy"] == 1
    assert report["call_counts"] == {
        "normal_llm_calls": 2,
        "recovery_calls": 1,
        "forced_final_answers": 0,
        "token_truncations": 1,
        "tool_calls": 1,
    }
    assert [p["max_output_tokens"] for p in client.calls] == [4096, 1024, 4096]
    assert (
        client.calls[0]["metadata"]["state_id"]
        == client.calls[1]["metadata"]["state_id"]
    )
    assert len(client.calls[1]["tools"]) == 11


def test_complete_truncated_action_needs_no_recovery(tmp_path, config, prepared):
    client = Client([geometry(truncated=True), raw(text="Final Answer: B")])
    report = execute(tmp_path, config, prepared, client)
    assert report["accuracy"] == 1 and report["call_counts"]["recovery_calls"] == 0
    assert report["call_counts"]["token_truncations"] == 1


def test_recovery_failure_ends_only_this_trajectory(tmp_path, config, prepared):
    client = Client(
        [raw(text="unfinished", truncated=True), raw(text="still reasoning")]
    )
    report = execute(tmp_path, config, prepared, client)
    assert report["status"] == "completed" and report["accuracy"] == 0
    assert report["response_status_counts"] == {"failed": 1}
    assert len(client.calls) == 2 and report["call_counts"]["tool_calls"] == 0


def test_tool_limit_forces_one_final_without_tools(tmp_path, config, prepared):
    client = Client([geometry(), raw(text="Final Answer: B")])
    report = execute(tmp_path, config, prepared, client, max_steps=1)
    assert report["accuracy"] == 1 and report["call_counts"]["tool_calls"] == 1
    assert report["call_counts"]["forced_final_answers"] == 1
    assert (
        "tools" not in client.calls[1] and client.calls[1]["max_output_tokens"] == 512
    )


def test_transport_failure_resumes_without_reexecuting_tool(tmp_path, config, prepared):
    client = Client([geometry(), RuntimeError("network temporarily down")])
    with pytest.raises(ModelRequestError):
        execute(tmp_path, config, prepared, client)
    assert not (tmp_path / "sat/results/deployment.json").exists()
    replay = Client([raw(text="Final Answer: B")])
    report = execute(tmp_path, config, prepared, replay)
    assert report["accuracy"] == 1 and len(replay.calls) == 1
    assert report["call_counts"]["tool_calls"] == 1


def test_guessed_private_path_is_recoverable_tool_observation(
    tmp_path, config, prepared
):
    client = Client(
        [
            raw(
                tool="draw",
                arguments={"image_uri": str(tmp_path / "data/sat/private.jsonl")},
            ),
            raw(text="Final Answer: B"),
        ]
    )
    report = execute(tmp_path, config, prepared, client)
    assert report["accuracy"] == 1
    observations = [
        i for i in client.calls[1]["input"] if i.get("type") == "function_call_output"
    ]
    assert "not an input image" in observations[0]["output"]
    assert "SECRET_LABEL" not in json.dumps(client.calls)


def test_wrong_model_or_failed_api_is_not_a_scored_failure():
    with pytest.raises(ValueError, match="Unexpected returned model"):
        parse_react_response(
            raw(text="Final Answer: B", model="gpt-5.4"), "gpt-5.4-mini"
        )
    with pytest.raises(RuntimeError, match="failed status"):
        parse_react_response({**raw(), "status": "failed"}, "gpt-5.4-mini")


def test_resume_rejects_changed_protocol(tmp_path, config, prepared):
    execute(tmp_path, config, prepared, Client([raw(text="Final Answer: B")]))
    with pytest.raises(ValueError, match="binding changed"):
        execute(tmp_path, config, prepared, Client([]), max_steps=49)
