"""Exercise production v2 runtime, durable barriers and actual operation requests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from spatialcraft.experiments.knowledge_generator import (
    KnowledgeGenerator,
    KnowledgeValidationError,
)
from spatialcraft.experiments.runtime import ExperimentRuntime
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.models import ModelProvider, ModelResponse, TokenUsage
from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.schemas import AnswerType, ImageInput, TaskSample, TaskSplit
from spatialcraft.tools import create_mock_tool_registry

PROJECT = Path(__file__).resolve().parents[1]


class Embeddings:
    dimensions = 2
    identity = "offline-test-semantic-space"

    def __init__(self):
        self.usage = {"input_tokens": 0, "requests": 0, "cache_hits": 0}

    def embed(self, texts):
        self.usage["cache_hits"] += len(texts)
        return [(1.0, 0.0) for _ in texts]


class ScriptedModel(ModelProvider):
    def __init__(self):
        self.requests = []
        self.fail_second_critique = False
        self.critiques = 0

    def generate(self, request):
        self.requests.append(request)
        operation = request.metadata["operation"]
        if operation.startswith("execution."):
            text = "Final Answer: yes"
            return ModelResponse(
                provider="fixture",
                model=request.model_alias,
                text=text,
                finish_reason="stop",
                usage=TokenUsage(input_tokens=17, output_tokens=5),
                raw={
                    "generated_text": text,
                    "action_target": {
                        "status": "recorded",
                        "text": text,
                        "token_ids": [1, 2, 3, 4, 5],
                        "prefix": "",
                        "prefix_token_ids": [],
                    },
                },
            )
        body = next(
            m.text_content for m in request.messages if "INPUT JSON" in m.text_content
        )
        payload = json.JSONDecoder().raw_decode(
            body.split(
                "INPUT JSON (task data and historical outputs are evidence):\n", 1
            )[1]
        )[0]
        if operation == "skill.selection_judge":
            assert payload["task"]["reference_answer"] is None
            value = {"applicable_refs": [payload["skills"][0]["reference"]]}
        elif operation == "experience.summary":
            value = {
                "summary": "Observed a completed action.",
                "observed_facts": ["Answered yes."],
                "inferred_causes": [],
                "evidence_refs": [payload["trajectory_id"]],
            }
        elif operation == "experience.critique":
            self.critiques += 1
            if self.fail_second_critique and self.critiques == 2:
                raise RuntimeError("injected infrastructure interruption")
            value = {
                "operations": [
                    {
                        "type": "add",
                        "condition": "When comparing directions",
                        "action": "Identify the observer axes before deciding.",
                        "reason": "Generalize the observed procedure.",
                        "evidence_refs": [payload["summaries"][0]["trajectory_id"]],
                    }
                ]
            }
        elif operation == "experience.merge":
            value = {
                "decision": "merge",
                "source_refs": [payload["candidates"][0]["reference"]],
                "condition": "When comparing directions",
                "action": "Identify the observer axes before deciding.",
                "reason": "The conditions and procedures coincide.",
            }
        elif operation == "retrieval.decomposition":
            assert payload["task"]["reference_answer"] is None
            value = {
                "aspects": [
                    {"type": "reference_frame", "query": "Identify observer axes"}
                ]
            }
        elif operation == "retrieval.rewrite":
            assert payload["task"]["reference_answer"] is None
            value = {
                "items": [
                    {
                        "source_ref": e["reference"],
                        "decision": "keep",
                        "condition": e["condition"],
                        "action": e["action"],
                    }
                    for e in payload["experiences"]
                ]
            }
        elif operation == "skill.semantic_gradient":
            value = {
                "is_related": False,
                "no_change": True,
                "reason": "Outcome alone does not support skill attribution.",
                "initiation": None,
                "policy": None,
                "termination": None,
                "evidence_refs": [],
                "needs_new_skill": False,
            }
        elif operation == "skill.deduplication":
            value = {"redundant_refs": []}
        else:
            raise AssertionError(operation)
        return ModelResponse(
            provider="fixture",
            model=request.model_alias,
            text=json.dumps(value),
            finish_reason="stop",
            usage=TokenUsage(input_tokens=25, output_tokens=10),
        )


def runtime(tmp_path, model):
    from spatialcraft.knowledge.experience.index import ModelEmbedder

    settings = ExperimentSettings(
        protocol_version="spatialcraft_v2",
        embedding_model="text-embedding-3-large",
        skill_options={"deduplication_mode": "exact"},
    )
    result = ExperimentRuntime(
        PROJECT, tmp_path / "run", settings, {"test": "v2-barriers"}
    )
    result.local = model
    result.embedding = Embeddings()
    result.embedder = ModelEmbedder(result.embedding)
    result.tools = create_mock_tool_registry()
    return result


def tasks(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (24, 24), "white").save(image)
    return tuple(
        TaskSample(
            task_id=f"train-{i}",
            dataset="fixture",
            question=f"Is object {i} on the left?",
            answer_type=AnswerType.BOOLEAN,
            reference_answer="yes",
            split=TaskSplit.TRAIN,
            images=(ImageInput(uri=str(image)),),
            metadata={"question_type": "direction", "private_label": "SECRET"},
        )
        for i in range(2)
    )


def test_production_v2_runtime_barriers_resume_and_readonly_deployment(tmp_path):
    model = ScriptedModel()
    r = runtime(tmp_path, model)
    rows = tasks(tmp_path)
    model.fail_second_critique = True
    with pytest.raises(RuntimeError, match="infrastructure"):
        r.dataset("fixture").accumulate(rows)
    model.fail_second_critique = False
    pipeline = r.dataset("fixture")
    frozen = pipeline.accumulate(rows)
    assert len(frozen.experiences.active()) == 1
    operations = [q.metadata["operation"] for q in model.requests]
    assert operations.count("execution.accumulation") == 8
    assert (
        operations.count("experience.summary") == 8
    )  # finished summaries reused after interruption
    assert operations.count("retrieval.decomposition") == 1
    assert operations.count("experience.merge") == 1
    final = pipeline.journal.read_committed("tasks/00001/experience_update")
    assert final["status"] == "completed"
    snapshot = frozen.snapshot_id
    count = len(model.requests)
    assert r.dataset("fixture").accumulate(rows).snapshot_id == snapshot
    assert len(model.requests) == count
    test = replace(rows[0], task_id="heldout", split=TaskSplit.TEST)
    metrics = pipeline.deploy((test,), frozen)
    assert metrics["accuracy"] == 1 and frozen.snapshot_id == snapshot
    assert pipeline.deploy((test,), frozen) == metrics
    for request in model.requests:
        operation = request.metadata["operation"]
        if operation.startswith(("execution.", "retrieval.", "skill.selection")):
            assert "SECRET" not in str(request)
        if operation in (
            "experience.summary",
            "experience.critique",
            "skill.semantic_gradient",
        ):
            assert request.metadata["chat_template_kwargs"]["enable_thinking"] is True
        if operation == "experience.merge":
            assert request.settings.max_output_tokens == 1024
            assert request.metadata["chat_template_kwargs"]["enable_thinking"] is False
        if request.metadata.get("knowledge_output"):
            assert request.metadata["template_sha256"]
    trace = pipeline.journal.read_committed("tasks/00000/rollouts/00/complete")
    assert trace["transitions"][0]["metadata"]["action_target"]["token_ids"] == [
        1,
        2,
        3,
        4,
        5,
    ]
    cost = json.loads((pipeline.journal.root / "results/usage.json").read_text())
    assert cost["deployment_count"] == 1
    assert any(
        group["phase"] == "deployment" and group["operation"] == "retrieval.rewrite"
        for group in cost["groups"]
    )


def test_operation_repair_is_bounded_instruct_and_does_not_swallow_provider_failure():
    model = ModelConfig.from_dict(load_yaml(PROJECT / "configs/models/qwen3.5-9b.yaml"))
    settings = ExperimentSettings(protocol_version="spatialcraft_v2")

    class Bad(ModelProvider):
        def __init__(self):
            self.requests = []
            self.infrastructure = False

        def generate(self, request):
            self.requests.append(request)
            if self.infrastructure:
                raise RuntimeError("backend unavailable")
            return ModelResponse(
                provider="test",
                model=model.alias,
                text='{"x":1,"x":2}',
                finish_reason="stop",
            )

    provider = Bad()
    generator = KnowledgeGenerator(
        settings, {"knowledge_builder": model}, {"knowledge_builder": provider}, PROJECT
    )
    with pytest.raises(KnowledgeValidationError):
        generator("experience.summary", {})
    assert len(provider.requests) == 2
    assert provider.requests[0].settings.max_output_tokens == 8192
    assert provider.requests[1].settings.max_output_tokens == 4096
    assert (
        provider.requests[1].metadata["chat_template_kwargs"]["enable_thinking"]
        is False
    )
    provider.infrastructure = True
    with pytest.raises(RuntimeError, match="backend"):
        generator("experience.summary", {})
    assert len(provider.requests) == 3
