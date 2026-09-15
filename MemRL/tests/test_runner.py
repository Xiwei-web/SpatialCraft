"""Real journaled spatial loops with deterministic offline MemRL providers."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from MemRL.reflection import ReflectionBuilder
from MemRL.runner import DatasetRunner, Settings, model_config
from spatialcraft.experiments.protocol import public_task
from spatialcraft.models import MessageRole, ModelResponse, TokenUsage
from spatialcraft.models.providers.openai_responses import OpenAIResponsesProvider
from spatialcraft.models.serialization import request_to_dict
from spatialcraft.schemas import AnswerType, ImageInput, TaskSample, TaskSplit
from spatialcraft.storage.atomic_io import sha256_file
from spatialcraft.tools.real import create_real_tool_registry

PROJECT = Path(__file__).resolve().parents[2]
SECRET = "PRIVATE_LABEL_AND_MASK_METADATA"
REFLECTION = "Verify the observer frame, then compare the relevant objects using current visual evidence."


class FakeActor:
    def __init__(self, *, recover=False):
        self.requests, self.recover = [], recover

    def factory(self, config, media):
        return self

    def generate(self, request):
        self.requests.append(request)
        query = next(
            message.text_content
            for message in request.messages
            if message.role is MessageRole.USER
        )
        recovering = (
            request.metadata.get("rollout_budget", {}).get("call_kind") == "recovery"
        )
        if self.recover and "ENV_FAIL" in query and not recovering:
            return ModelResponse(
                provider="fixture",
                model=request.model_alias,
                text=None,
                finish_reason="length",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        answer = "no" if "ENV_FAIL" in query else "yes"
        return ModelResponse(
            provider="fixture",
            model=request.model_alias,
            text=f"Final Answer: {answer}",
            finish_reason="completed",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


class FakeReflection:
    def __init__(self, *, invalid=None):
        self.requests, self.invalid = [], invalid

    def factory(self, config, media):
        return self

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(
            provider="fixture",
            model=request.model_alias,
            text="" if self.invalid == "empty" else REFLECTION,
            finish_reason="incomplete" if self.invalid == "length" else "completed",
            usage=TokenUsage(input_tokens=20, output_tokens=7),
        )


class FakeEmbedder:
    identity = "fixture-spatial-intent-vectors:2"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(tuple(texts))
        vectors = []
        for text in texts:
            if "ENV_FAIL" in text:
                vectors.append([1.0, 0.0])
            elif "ENV_SUCCESS_1" in text:
                vectors.append([0.9, 0.1])
            elif "ENV_SUCCESS_2" in text:
                vectors.append([0.0, 1.0])
            elif "ALTERNATE_HELDOUT" in text:
                vectors.append([-1.0, 0.0])
            else:
                vectors.append([0.1, 0.9])
        return vectors


def setup(
    tmp_path, *, variant="R", invalid_reflection=None, recover=False, settings=None
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    private, media = [], {}
    for index, name in enumerate(
        ("ENV_FAIL", "ENV_SUCCESS_1", "ENV_SUCCESS_2", "HELDOUT_QUERY")
    ):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (16, 16), (index * 50, 20, 30)).save(path)
        media[str(path)] = sha256_file(path)
        private.append(
            TaskSample(
                dataset="sat",
                task_id=f"task-{index}",
                question=f"{name}: is the object left?",
                answer_type=AnswerType.BOOLEAN,
                reference_answer="yes",
                images=(
                    ImageInput(
                        uri=str(path),
                        media_type="image/png",
                        metadata={"label": SECRET},
                    ),
                ),
                metadata={
                    "question_type": "direction",
                    "reference_answer_text": SECRET,
                    "experiment_split": "environment" if index < 3 else "deployment",
                },
                split=TaskSplit.TRAIN if index < 3 else TaskSplit.TEST,
            )
        )
    public = [TaskSample.from_dict(public_task(task)) for task in private]
    actor, reflection, embedder = (
        FakeActor(recover=recover),
        FakeReflection(invalid=invalid_reflection),
        FakeEmbedder(),
    )
    settings = settings or Settings(
        similarity_threshold=-1.0,
        top_k=1,
        candidate_k=10,
        utility_weight=0.0,
        max_steps=3,
    )
    config = model_config(PROJECT, settings)
    runner = DatasetRunner(
        tmp_path / "run",
        "sat",
        {"media": media},
        settings,
        config,
        {"fixture": True},
        variant=variant,
        embedder=embedder,
        builder=ReflectionBuilder(config, variant, reflection.factory),
        registry=create_real_tool_registry(),
        provider_factory=actor.factory,
    )
    return SimpleNamespace(
        runner=runner,
        actor=actor,
        reflection=reflection,
        embedder=embedder,
        environment=(tuple(public[:3]), tuple(private[:3])),
        deployment=(tuple(public[3:]), tuple(private[3:])),
    )


def call_counts(fixture):
    return (
        len(fixture.actor.requests),
        len(fixture.reflection.requests),
        len(fixture.embedder.calls),
    )


@pytest.mark.parametrize("variant", ["R", "GT"])
def test_online_environment_learns_from_both_outcomes_and_deployment_is_frozen(
    tmp_path, variant
):
    fixture = setup(tmp_path, variant=variant)
    runner = fixture.runner
    report = runner.run("all", fixture.environment, fixture.deployment)
    assert report["accuracy"] == 1 and report["memory_count"] == 3
    rows = runner.journal.read_committed("environment_complete")["rows"]
    assert [row["reward"] for row in rows] == [0, 1, 1]
    assert len(fixture.reflection.requests) == 3
    assert [
        json.loads(request.messages[-1].content[0].text)["scalar_reward"]
        for request in fixture.reflection.requests
    ] == [0, 1, 1]
    events = [
        runner.journal.read_committed(f"environment/000/{i:05d}/000/memory_update")
        for i in range(3)
    ]
    assert events[0]["selected_ids"] == [] and events[0]["updated_utilities"] == []
    assert all(
        event["new_record"]["q_value"] == 0.0 and event["new_record"]["visits"] == 0
        for event in events
    )
    # The failed first task is admitted; its memory influences the next actor call.
    assert events[1]["selected_ids"] == [events[0]["new_record"]["memory_id"]]
    assert events[2]["selected_ids"] == [events[1]["new_record"]["memory_id"]]
    assert all(
        event["updated_utilities"][0]["q_value"] == pytest.approx(0.3)
        for event in events[1:]
    )
    for index, event in enumerate(events):
        selection = runner.journal.read_committed(
            f"environment/000/{index:05d}/000/retrieval"
        )
        assert (
            event["selected_ids"]
            == selection["selected_ids"]
            == fixture.actor.requests[index].metadata["memory_ids"]
        )
        assert [item["memory_id"] for item in event["updated_utilities"]] == event[
            "selected_ids"
        ]
    frozen = runner.journal.read_committed("frozen_memory")
    assert [(item["q_value"], item["visits"]) for item in frozen["records"]] == [
        (0.3, 1),
        (0.3, 1),
        (0.0, 0),
    ]
    memory_message = next(
        message.text_content
        for message in fixture.actor.requests[1].messages
        if message.text_content.startswith("MemRL procedural memories")
    )
    assert REFLECTION in memory_message and "ENV_FAIL" in memory_message
    for request in fixture.actor.requests:
        payload = json.dumps(request_to_dict(request, identity=False))
        assert SECRET not in payload and '"correct_answer"' not in payload
        assert '"reference_answer"' not in payload
    for request in fixture.reflection.requests:
        payload = json.loads(request.messages[-1].content[0].text)
        assert ("correct_answer" in payload) is (variant == "GT")
        assert SECRET not in json.dumps(request_to_dict(request))
    frozen_path = runner.root / "memory/frozen.json"
    before_bytes = frozen_path.read_bytes()
    before_calls = call_counts(fixture)
    again = runner.run("all", fixture.environment, fixture.deployment)
    assert call_counts(fixture) == before_calls == (4, 3, 4)
    assert again["memory_snapshot"] == report["memory_snapshot"]
    assert runner.journal.read_committed("frozen_memory") == frozen
    assert frozen_path.read_bytes() == before_bytes
    assert not list(
        (runner.journal.root / "stages/deployment").glob("**/reflection/result.json")
    )
    assert not list(
        (runner.journal.root / "stages/deployment").glob("**/memory_update/result.json")
    )
    environment_report = json.loads(
        (runner.root / "results/environment.json").read_text()
    )
    assert (
        environment_report["reflection_calls"] == 3
        and environment_report["utility_updates"] == 2
    )
    assert environment_report["usage"]["input_tokens"] == 30
    assert environment_report["usage"]["reasoning_tokens"] is None
    assert environment_report["reflection_usage"]["input_tokens"] == 60
    assert environment_report["reflection_usage"]["reasoning_tokens"] is None
    assert (
        report["usage"]["input_tokens"] == 10
        and report["usage"]["cached_input_tokens"] is None
    )


def test_deployment_and_invalid_stage_barriers_precede_all_calls(tmp_path):
    fixture = setup(tmp_path)
    with pytest.raises(FileNotFoundError):
        fixture.runner.run("deployment", fixture.environment, fixture.deployment)
    with pytest.raises(ValueError, match="Unknown MemRL stage"):
        fixture.runner.run("build", fixture.environment, fixture.deployment)
    assert call_counts(fixture) == (0, 0, 0)
    fixture.runner.run("environment", fixture.environment, fixture.deployment)
    assert call_counts(fixture) == (3, 3, 3)
    frozen = fixture.runner.journal.read_committed("frozen_memory")
    fixture.runner.run("deployment", fixture.environment, fixture.deployment)
    assert call_counts(fixture) == (4, 3, 4)
    assert fixture.runner.journal.read_committed("frozen_memory") == frozen


@pytest.mark.parametrize("variant", ["R", "GT"])
@pytest.mark.parametrize("invalid", ["empty", "length"])
def test_failed_reflection_commits_cost_before_rejecting_and_cannot_silently_retry(
    tmp_path, variant, invalid
):
    fixture = setup(tmp_path, variant=variant, invalid_reflection=invalid)
    for _ in range(2):
        with pytest.raises(ValueError, match="reflection did not produce"):
            fixture.runner.run("all", fixture.environment, fixture.deployment)
        fixture.reflection.invalid = None
    assert call_counts(fixture) == (1, 1, 3)
    committed = fixture.runner.journal.read_committed(
        "environment/000/00000/000/reflection"
    )
    assert committed["usage"]["input_tokens"] == 20
    assert committed["usage"]["output_tokens"] == 7
    assert not (fixture.runner.root / "memory/frozen.json").exists()
    assert not (
        fixture.runner.root / "stages/environment_complete/result.json"
    ).exists()
    assert not (
        fixture.runner.root
        / "stages/environment/000/00000/000/memory_update/result.json"
    ).exists()


def test_medium_budget_is_preserved_for_actor_reflection_and_actual_recovery(tmp_path):
    fixture = setup(tmp_path, recover=True)
    fixture.runner.run("all", fixture.environment, fixture.deployment)
    recovery = [
        request
        for request in fixture.actor.requests
        if request.metadata.get("rollout_budget", {}).get("call_kind") == "recovery"
    ]
    assert len(recovery) == 1
    provider = OpenAIResponsesProvider(fixture.runner.config)
    for request in fixture.actor.requests + fixture.reflection.requests:
        payload = provider.build_payload(request)
        assert payload["model"] == "gpt-5.4-mini"
        assert payload["reasoning"] == {"effort": "medium"}
        assert payload["max_output_tokens"] == 16384
        assert not {"temperature", "top_p", "seed"}.intersection(payload)
    trajectory = fixture.runner.journal.read_committed(
        "environment/000/00000/000/trajectory"
    )
    assert [event["kind"] for event in trajectory["metadata"]["generation_events"]] == [
        "normal",
        "recovery",
    ]
    environment_report = json.loads(
        (fixture.runner.root / "results/environment.json").read_text()
    )
    assert environment_report["usage"]["input_tokens"] == 40


def test_threshold_calibration_reads_environment_queries_only(tmp_path):
    settings = Settings(similarity_threshold=None, top_k=1, max_steps=3)
    original = setup(tmp_path / "original", settings=settings)
    changed = setup(tmp_path / "changed", settings=settings)
    heldout_private = replace(
        changed.deployment[1][0],
        question="ALTERNATE_HELDOUT: unrelated question?",
        reference_answer="no",
    )
    changed.deployment = (
        (TaskSample.from_dict(public_task(heldout_private)),),
        (heldout_private,),
    )
    for fixture in (original, changed):
        fixture.runner.run("environment", fixture.environment, fixture.deployment)
        assert fixture.embedder.calls == [
            (task.question,) for task in fixture.environment[0]
        ]
    left = original.runner.journal.read_committed("calibration/threshold")
    right = changed.runner.journal.read_committed("calibration/threshold")
    assert left == right
    assert left["source"] == "environment_public_pairwise_cosine_quantile"
    assert left["task_count"] == 3 and left["pair_count"] == 3
    changed.runner.run("deployment", changed.environment, changed.deployment)
    assert changed.embedder.calls[-1] == (heldout_private.question,)
    assert changed.runner.journal.read_committed("calibration/threshold") == right


def test_invalid_public_private_pairing_is_rejected_before_environment_cost(tmp_path):
    fixture = setup(tmp_path)
    changed_private = replace(
        fixture.deployment[1][0], question="Mismatched private question"
    )
    fixture.deployment = (fixture.deployment[0], (changed_private,))
    with pytest.raises(ValueError, match="pairing mismatch"):
        fixture.runner.run("all", fixture.environment, fixture.deployment)
    assert call_counts(fixture) == (0, 0, 0)


@pytest.mark.parametrize("variant", ["R", "GT"])
@pytest.mark.parametrize("stage_suffix", ["reflection", "memory_update"])
def test_resume_after_committed_learning_step_has_no_double_calls_or_utility_updates(
    tmp_path, variant, stage_suffix, monkeypatch
):
    fixture = setup(tmp_path, variant=variant)
    journal = fixture.runner.journal
    execute = journal.execute
    interrupted_key = "environment/000/00001/000/" + stage_suffix
    crashed = False

    def crash_after_commit(key, inputs, operation):
        nonlocal crashed
        result = execute(key, inputs, operation)
        if key == interrupted_key and not crashed:
            crashed = True
            raise RuntimeError("fixture crash after durable learning stage")
        return result

    monkeypatch.setattr(journal, "execute", crash_after_commit)
    with pytest.raises(RuntimeError, match="fixture crash"):
        fixture.runner.run("all", fixture.environment, fixture.deployment)
    assert crashed and journal.read_committed(interrupted_key)
    assert call_counts(fixture) == (2, 2, 3)
    assert not (fixture.runner.root / "memory/frozen.json").exists()
    fixture.runner.run("all", fixture.environment, fixture.deployment)
    assert call_counts(fixture) == (4, 3, 4)
    frozen = journal.read_committed("frozen_memory")
    assert [(item["q_value"], item["visits"]) for item in frozen["records"]] == [
        (0.3, 1),
        (0.3, 1),
        (0.0, 0),
    ]
    assert len(frozen["records"]) == 3
    assert len({item["memory_id"] for item in frozen["records"]}) == 3
    rows = journal.read_committed("environment_complete")["rows"]
    assert [row["reward"] for row in rows] == [0, 1, 1]
    assert [row["new_memory_id"] for row in rows] == [
        item["memory_id"] for item in frozen["records"]
    ]


@pytest.mark.parametrize("variant", ["R", "GT"])
def test_unparseable_actor_output_still_becomes_failure_reflection_and_memory(
    tmp_path, variant, monkeypatch
):
    fixture = setup(tmp_path, variant=variant)
    original_generate = fixture.actor.generate
    bare_output = "[[0.64,0.38]]"

    def unparseable_first_task(request):
        query = next(
            message.text_content
            for message in request.messages
            if message.role is MessageRole.USER
        )
        if "ENV_FAIL" not in query:
            return original_generate(request)
        fixture.actor.requests.append(request)
        return ModelResponse(
            provider="fixture",
            model=request.model_alias,
            text=bare_output,
            finish_reason="completed",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    monkeypatch.setattr(fixture.actor, "generate", unparseable_first_task)
    fixture.runner.run("all", fixture.environment, fixture.deployment)
    trajectory = fixture.runner.journal.read_committed(
        "environment/000/00000/000/trajectory"
    )
    assert trajectory["transitions"] == []
    assert trajectory["final_answer"] is None
    rows = fixture.runner.journal.read_committed("environment_complete")["rows"]
    assert rows[0]["reward"] == 0.0
    evidence = json.loads(fixture.reflection.requests[0].messages[-1].content[0].text)
    assert evidence["trajectory"] == []
    assert evidence["model_output"]["text"] == bare_output
    assert evidence["model_output"]["final_answer"] is None
    assert evidence["scalar_reward"] == 0.0
    frozen = fixture.runner.journal.read_committed("frozen_memory")
    assert len(frozen["records"]) == 3
    assert frozen["records"][0]["source_task_id"] == fixture.environment[0][0].task_id
    assert frozen["records"][0]["reflection"] == REFLECTION
