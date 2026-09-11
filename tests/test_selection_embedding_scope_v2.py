"""Selection embeddings retain their operation without changing model cache identity."""

import json
from dataclasses import replace

import pytest
from test_runtime_v2 import ScriptedModel, runtime, tasks

from spatialcraft.experiments.journal import RunJournal
from spatialcraft.experiments.usage import ScopedEmbedder
from spatialcraft.knowledge.experience.index import ModelEmbedder
from spatialcraft.models.serialization import request_to_dict
from spatialcraft.schemas import TaskSplit


class InterruptedExecutor(ScriptedModel):
    def __init__(self):
        super().__init__()
        self.interrupt_operation = None
        self.executor_attempts = []

    def generate(self, request):
        operation = request.metadata["operation"]
        if operation.startswith("execution."):
            self.executor_attempts.append(request_to_dict(request, identity=False))
            if operation == self.interrupt_operation:
                self.interrupt_operation = None
                raise RuntimeError("fixture executor interruption after selection")
        return super().generate(request)


def events(root):
    return [json.loads(path.read_text()) for path in (root / "usage").glob("*.json")]


@pytest.mark.parametrize("interrupted_phase", ["accumulation", "deployment"])
def test_selection_embedding_operation_and_executor_identity_survive_runtime_resume(
    tmp_path, interrupted_phase
):
    model = InterruptedExecutor()
    instance = runtime(tmp_path, model)
    training = tasks(tmp_path)[:1]
    heldout = (replace(training[0], task_id="heldout-selection", split=TaskSplit.TEST),)
    pipeline = instance.dataset("fixture")
    if interrupted_phase == "accumulation":
        model.interrupt_operation = "execution.accumulation"
        with pytest.raises(RuntimeError, match="after selection"):
            pipeline.accumulate(training)
        failed = model.executor_attempts[-1]
        committed_selection = pipeline.journal.read_committed(
            "tasks/00000/rollouts/00/execution/steps/0000/selection"
        )
        assert committed_selection["active_skill"] is not None
        pipeline = instance.dataset("fixture")
    frozen = pipeline.accumulate(training)
    if interrupted_phase == "deployment":
        model.interrupt_operation = "execution.deployment"
        with pytest.raises(RuntimeError, match="after selection"):
            pipeline.deploy(heldout, frozen)
        failed = model.executor_attempts[-1]
        committed_selection = pipeline.journal.read_committed(
            "deployment/00000/execution/steps/0000/selection"
        )
        assert committed_selection["active_skill"] is not None
        pipeline = instance.dataset("fixture")
    result = pipeline.deploy(heldout, frozen)
    assert result["accuracy"] == 1

    matching_attempts = [
        wire
        for wire in model.executor_attempts
        if wire["metadata"]["task_id"] == failed["metadata"]["task_id"]
        and wire["metadata"]["rollout_index"] == failed["metadata"]["rollout_index"]
    ]
    assert len(matching_attempts) == 2
    assert matching_attempts[0] == matching_attempts[1]
    assert all(
        "embedding_operation" not in wire["metadata"]
        for wire in model.executor_attempts
    )

    ledger = events(pipeline.journal.root)
    selection_embeddings = [
        event
        for event in ledger
        if event["kind"] == "embedding" and event["operation"] == "skill.selection"
    ]
    assert len(selection_embeddings) == 5
    assert (
        sum(
            event["phase"] == "accumulation_execution" for event in selection_embeddings
        )
        == 4
    )
    assert sum(event["phase"] == "deployment" for event in selection_embeddings) == 1
    assert {
        event["rollout_index"]
        for event in selection_embeddings
        if event["phase"] == "accumulation_execution"
    } == {0, 1, 2, 3}
    assert all(
        event["rollout_prefix"] and event["snapshot_id"]
        for event in selection_embeddings
    )
    assert not any(
        event["kind"] == "embedding"
        and str(event.get("operation")).startswith("execution.")
        for event in ledger
    )
    assert any(
        event["kind"] == "embedding"
        and event["operation"] == "retrieval.index_and_query"
        for event in ledger
    )
    judge_calls = [
        q for q in model.requests if q.metadata["operation"] == "skill.selection_judge"
    ]
    assert (
        len(judge_calls) == 5
    )  # The cached selection is not judged or embedded again.
    executor_events = [
        event
        for event in ledger
        if event["kind"] == "generation"
        and event["operation"] == failed["metadata"]["operation"]
        and event["task_id"] == failed["metadata"]["task_id"]
        and event["rollout_index"] == failed["metadata"]["rollout_index"]
    ]
    assert {event["status"] for event in executor_events} == {"completed", "failed"}
    assert len({event["request_sha256"] for event in executor_events}) == 1

    counts = len(model.requests), len(selection_embeddings)
    assert pipeline.accumulate(training).snapshot_id == frozen.snapshot_id
    assert pipeline.deploy(heldout, frozen) == result
    assert len(model.requests) == counts[0]
    assert (
        len(
            [
                e
                for e in events(pipeline.journal.root)
                if e["kind"] == "embedding" and e["operation"] == "skill.selection"
            ]
        )
        == counts[1]
    )


def test_fixed_selection_operation_survives_embedding_failure_without_mutating_scope(
    tmp_path,
):
    class BrokenEmbedding:
        identity = "fixture-embedding"
        dimensions = 2

        def __init__(self):
            self.usage = {"input_tokens": 0, "requests": 0, "cache_hits": 0}

        def embed(self, texts):
            raise RuntimeError("fixture embedding unavailable")

    scope = {
        "phase": "deployment",
        "task_id": "t",
        "operation": "execution.deployment",
        "embedding_operation": "outer-operation",
        "rollout_index": 0,
    }
    before = dict(scope)
    wrapped = ScopedEmbedder(
        ModelEmbedder(BrokenEmbedding()),
        RunJournal(tmp_path, {"fixture": True}),
        lambda: scope,
        operation="skill.selection",
    )
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        wrapped.embed(["frame query"])
    assert scope == before
    (failed,) = events(tmp_path)
    assert failed["operation"] == "skill.selection"
    assert failed["embedding_operation"] == "skill.selection"
    assert failed["phase"] == "deployment"
    assert failed["status"] == "failed"
    assert failed["input_tokens"] is None and failed["requests"] is None
