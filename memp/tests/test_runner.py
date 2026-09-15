"""Offline MemP stages exercised through the actual spatial execution/journal path."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from memp.runner import DatasetRunner, ScriptBuilder, Settings, model_config
from spatialcraft.experiments.protocol import public_task
from spatialcraft.models import (
    ModelProvider,
    ModelResponse,
    ResponseToolCall,
    TokenUsage,
)
from spatialcraft.models.serialization import request_to_dict
from spatialcraft.schemas import AnswerType, ImageInput, TaskSample, TaskSplit
from spatialcraft.storage.atomic_io import sha256_file
from spatialcraft.tools.real import create_real_tool_registry

PROJECT = Path(__file__).resolve().parents[2]
SCRIPT = "1. Identify the current observer frame.\n2. Verify current spatial evidence before answering."
SECRET = "PRIVATE_LABEL_AND_REFERENCE_METADATA"


class FakeActor(ModelProvider):
    def __init__(self):
        self.requests = []
        self.old_uri = None

    def generate(self, request):
        self.requests.append(request)
        if (
            self.old_uri
            and request.metadata.get("memory_ids")
            and request.metadata["step_index"] == 0
        ):
            return ModelResponse(
                provider="fixture",
                model=request.model_alias,
                finish_reason="completed",
                tool_calls=(
                    ResponseToolCall(
                        name="draw", arguments={"image_uri": self.old_uri}
                    ),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        query_text = "\n".join(message.text_content for message in request.messages)
        answer = "no" if "ENV_FAIL" in query_text else "yes"
        return ModelResponse(
            provider="fixture",
            model=request.model_alias,
            text=f"Final Answer: {answer}",
            finish_reason="completed",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    def factory(self, config, media):
        return self


class FakeScriptProvider(ModelProvider):
    def __init__(self, invalid=None):
        self.requests, self.invalid = [], invalid

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(
            provider="fixture",
            model="gpt-5.4-mini" if self.invalid == "model" else request.model_alias,
            text="" if self.invalid == "empty" else SCRIPT,
            finish_reason="incomplete" if self.invalid == "truncated" else "completed",
            tool_calls=(ResponseToolCall(name="invented", arguments={}),)
            if self.invalid == "tool"
            else (),
            usage=TokenUsage(input_tokens=20, output_tokens=7),
        )


class FakeEmbedder:
    identity = "fixture-query-embedding:2"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(tuple(texts))
        return [[1.0, 0.0] for _ in texts]


def setup(tmp_path, *, memory_format="proceduralization", invalid_script=None):
    private = []
    media = {}
    for index, name in enumerate(("ENV_SUCCESS", "ENV_FAIL", "HELDOUT_QUERY")):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (16, 16), (index * 80, 20, 30)).save(path)
        media[str(path)] = sha256_file(path)
        private.append(
            TaskSample(
                dataset="sat",
                task_id=f"task-{index}",
                question=f"{name}: is the object left?",
                answer_type=AnswerType.BOOLEAN,
                reference_answer="yes",
                images=(ImageInput(uri=str(path), metadata={"label": SECRET}),),
                metadata={
                    "question_type": "direction",
                    "reference_answer_text": SECRET,
                },
                split=TaskSplit.TRAIN if index < 2 else TaskSplit.TEST,
            )
        )
    public = [TaskSample.from_dict(public_task(task)) for task in private]
    environment = (tuple(public[:2]), tuple(private[:2]))
    deployment = (tuple(public[2:]), tuple(private[2:]))
    actor, script_provider, embedder = (
        FakeActor(),
        FakeScriptProvider(invalid_script),
        FakeEmbedder(),
    )
    settings = Settings(memory_format=memory_format, max_steps=3)
    config = model_config(PROJECT, settings)
    runner = DatasetRunner(
        tmp_path / "run",
        "sat",
        {"media": media},
        settings,
        config,
        {"fixture": True},
        embedder=embedder,
        builder=ScriptBuilder(config, script_provider),
        registry=create_real_tool_registry(),
        provider_factory=actor.factory,
    )
    return SimpleNamespace(
        runner=runner,
        actor=actor,
        script_provider=script_provider,
        embedder=embedder,
        environment=environment,
        deployment=deployment,
    )


@pytest.mark.parametrize("memory_format", ["trajectory", "script", "proceduralization"])
def test_environment_build_frozen_deployment_and_replay(memory_format, tmp_path):
    fixture = setup(tmp_path, memory_format=memory_format)
    runner = fixture.runner
    report = runner.run("all", fixture.environment, fixture.deployment)
    complete = runner.journal.read_committed("environment_complete")
    assert [row["reward"] for row in complete["rows"]] == [1, 0]
    assert report["memory_count"] == 1 and report["accuracy"] == 1
    frozen = runner.journal.read_committed("frozen_memory")
    assert len(frozen["records"]) == 1
    assert frozen["records"][0]["source_task_id"] == fixture.environment[0][0].task_id
    assert frozen["records"][0]["trajectory"][0]["final_answer"] == "Final Answer: yes"
    assert len(fixture.script_provider.requests) == (
        0 if memory_format == "trajectory" else 1
    )
    if fixture.script_provider.requests:
        body = json.loads(fixture.script_provider.requests[0].messages[-1].text_content)
        assert body["source_query"] == fixture.environment[0][0].question
        assert body["successful_trajectory"][0]["final_answer"] == "Final Answer: yes"
    assert fixture.embedder.calls == [
        (fixture.environment[0][0].question,),
        (fixture.deployment[0][0].question,),
    ]
    assert not any(
        request.metadata["memory_ids"] for request in fixture.actor.requests[:2]
    )
    final_request = fixture.actor.requests[-1]
    memory_text = next(
        message.text_content
        for message in final_request.messages
        if "Historical MemP examples" in message.text_content
    )
    (item,) = json.loads(memory_text.split("\n", 1)[1])["memories"]
    assert ("trajectory" in item) is (memory_format != "script")
    assert ("script" in item) is (memory_format != "trajectory")
    if memory_format == "proceduralization":
        assert (
            item["trajectory"] == frozen["records"][0]["trajectory"]
            and item["script"] == SCRIPT
        )
    for request in fixture.actor.requests + fixture.script_provider.requests:
        text = json.dumps(request_to_dict(request, identity=False))
        assert SECRET not in text
        assert '"reference_answer"' not in text
    # Missing optional provider usage stays unknown at trajectory and report level.
    assert all(row["usage"]["reasoning_tokens"] is None for row in complete["rows"])
    assert report["usage"]["input_tokens"] == 10
    assert report["usage"]["reasoning_tokens"] is None
    summary = json.loads((runner.root / "memory/build_summary.json").read_text())
    assert summary["script_calls"] == (0 if memory_format == "trajectory" else 1)
    assert summary["script_usage"]["input_tokens"] == (
        0 if memory_format == "trajectory" else 20
    )
    if memory_format != "trajectory":
        assert summary["script_usage"]["reasoning_tokens"] is None
    frozen_path = runner.root / "memory/frozen.json"
    frozen_bytes = frozen_path.read_bytes()
    calls = (
        len(fixture.actor.requests),
        len(fixture.script_provider.requests),
        len(fixture.embedder.calls),
    )
    repeated = runner.run("all", fixture.environment, fixture.deployment)
    assert (
        repeated["accuracy"] == 1
        and repeated["memory_snapshot"] == report["memory_snapshot"]
    )
    assert (
        len(fixture.actor.requests),
        len(fixture.script_provider.requests),
        len(fixture.embedder.calls),
    ) == calls
    assert runner.journal.read_committed("frozen_memory") == frozen
    assert frozen_path.read_bytes() == frozen_bytes


def test_stage_barriers_precede_any_builder_or_embedding_calls(tmp_path):
    fixture = setup(tmp_path)
    for stage in ("build", "deployment"):
        with pytest.raises((FileNotFoundError, ValueError)):
            fixture.runner.run(stage, fixture.environment, fixture.deployment)
    assert (
        not fixture.actor.requests
        and not fixture.script_provider.requests
        and not fixture.embedder.calls
    )
    assert (
        fixture.runner.run("environment", fixture.environment, fixture.deployment)[
            "stage"
        ]
        == "environment"
    )
    assert len(fixture.actor.requests) == 2
    assert not fixture.script_provider.requests and not fixture.embedder.calls
    fixture.runner.run("build", fixture.environment, fixture.deployment)
    assert (
        len(fixture.actor.requests) == 2 and len(fixture.script_provider.requests) == 1
    )
    assert len(fixture.embedder.calls) == 1
    report = fixture.runner.run("deployment", fixture.environment, fixture.deployment)
    assert report["accuracy"] == 1 and len(fixture.actor.requests) == 3
    before = (
        len(fixture.actor.requests),
        len(fixture.script_provider.requests),
        len(fixture.embedder.calls),
    )
    with pytest.raises(ValueError, match="Unknown MemP stage"):
        fixture.runner.run("typo", fixture.environment, fixture.deployment)
    assert before == (
        len(fixture.actor.requests),
        len(fixture.script_provider.requests),
        len(fixture.embedder.calls),
    )


@pytest.mark.parametrize("invalid", ["truncated", "empty", "model", "tool"])
def test_invalid_paid_script_response_remains_durable_without_embedding_or_silent_retry(
    tmp_path, invalid
):
    fixture = setup(tmp_path, invalid_script=invalid)
    fixture.runner.run("environment", fixture.environment, fixture.deployment)
    with pytest.raises(ValueError):
        fixture.runner.run("build", fixture.environment, fixture.deployment)
    wire = fixture.runner.journal.read_committed("build/00000/000/script")
    assert wire["usage"]["input_tokens"] == 20 and wire["usage"]["output_tokens"] == 7
    if invalid == "truncated":
        assert wire["finish_reason"] == "incomplete"
    assert len(fixture.script_provider.requests) == 1 and not fixture.embedder.calls
    assert not (fixture.runner.root / "memory/frozen.json").exists()
    fixture.script_provider.invalid = None
    with pytest.raises(ValueError):
        fixture.runner.run("build", fixture.environment, fixture.deployment)
    assert len(fixture.script_provider.requests) == 1 and not fixture.embedder.calls


def test_historical_image_cannot_be_reused_as_current_tool_input(tmp_path):
    fixture = setup(tmp_path)
    fixture.actor.old_uri = fixture.environment[0][0].images[0].uri
    report = fixture.runner.run("all", fixture.environment, fixture.deployment)
    assert report["accuracy"] == 1
    trajectory = fixture.runner.journal.read_committed("deployment/00000/trajectory")
    result = trajectory["transitions"][0]["tool_results"][0]
    assert result["error_type"] == "argument_validation"
    assert "not an input image" in result["error_message"]
    assert len(trajectory["transitions"]) == 2
    assert "not an input image" in str(request_to_dict(fixture.actor.requests[-1]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "other"},
        {"reasoning_effort": "bad"},
        {"memory_format": "memp_reflection"},
        {"embedding_model": "unknown"},
        {"max_output_tokens": 0},
        {"environment_rollouts": True},
        {"max_steps": -1},
        {"top_k": 0},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": True},
        {"temperature": "0"},
    ],
)
def test_invalid_settings_are_rejected_before_running(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
