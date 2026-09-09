from __future__ import annotations

import base64

from spatialcraft.datasets import (
    ImageMaterializer,
    SplitManager,
    create_default_registry,
)
from spatialcraft.datasets.normalizer import parse_inline_choices
from spatialcraft.models import ModelProvider, ModelResponse
from spatialcraft.schemas import AgentAction, AnswerType, TaskSample, TaskSplit
from spatialcraft.verification import LLMJudgeVerifier, VerifierRouter


def _task(
    answer_type: AnswerType,
    reference: object,
    *,
    choices: tuple[str, ...] = (),
    metadata: dict[str, object] | None = None,
) -> TaskSample:
    return TaskSample(
        dataset="test",
        question="Where is the object?",
        answer_type=answer_type,
        reference_answer=reference,
        choices=choices,
        metadata=metadata or {},
    )


def test_erqa_choice_formats() -> None:
    prompts = (
        "Question? Choices: A. A. B. B. C. C. D. D. Please answer directly.",
        "Question? A) [10 20] B) [20 30] C) [30 40] D) [40 50] Please answer.",
        "Question? You have the following options. A: left B: right C: up D: down",
    )
    for prompt in prompts:
        question, labels, choices = parse_inline_choices(prompt)
        assert question
        assert labels == ("A", "B", "C", "D")
        assert len(choices) == 4


def test_default_registry_exposes_five_adapters(tmp_path) -> None:
    registry = create_default_registry(tmp_path)
    assert registry.names() == (
        "erqa",
        "omni3d",
        "robospatial",
        "sat",
        "viewspatial",
    )
    assert registry.create("robo-spatial").dataset_name == "robospatial"


def test_image_materialization_is_content_addressed(tmp_path) -> None:
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    )
    materializer = ImageMaterializer(tmp_path)
    first = materializer.from_bytes(
        png, dataset="test", source_id="one", index=0, suggested_path="one.png"
    )
    second = materializer.from_bytes(
        png, dataset="test", source_id="two", index=0, suggested_path="two.png"
    )
    assert first.uri == second.uri
    assert first.sha256 == second.sha256


def test_split_manager_is_deterministic_and_group_safe() -> None:
    samples = tuple(
        TaskSample(
            dataset="test",
            question=f"question {index}",
            source_id=f"group-{index // 2}",
            metadata={"question_type": "even" if index % 2 == 0 else "odd"},
        )
        for index in range(10)
    )
    manager = SplitManager(seed=17)
    assert manager.sample(samples, 4) == manager.sample(samples, 4)
    partitions = manager.partition_tasks(samples, train=0.6, validation=0.2, test=0.2)
    assigned: dict[str, TaskSplit] = {}
    for split, tasks in partitions.items():
        for task in tasks:
            key = task.source_id or ""
            assert assigned.setdefault(key, split) is split
            assert task.split is split


def test_router_converts_supported_outputs_to_rewards() -> None:
    router = VerifierRouter()
    cases = (
        (_task(AnswerType.FREE_FORM, ["left", "west"]), "West.", 1.0),
        (_task(AnswerType.BOOLEAN, "yes"), AgentAction.final("TRUE"), 1.0),
        (
            _task(
                AnswerType.MULTIPLE_CHOICE,
                "B",
                choices=("red", "blue", "green"),
            ),
            "Answer: B",
            1.0,
        ),
        (_task(AnswerType.NUMERIC, "50%"), "0.5", 1.0),
        (_task(AnswerType.SPATIAL_RELATION, "in front of"), "front", 1.0),
        (
            _task(
                AnswerType.POINTING,
                "[(0.5, 0.5)]",
                metadata={"point_tolerance": 0.02},
            ),
            "[[0.51, 0.5]]",
            1.0,
        ),
        (_task(AnswerType.STRUCTURED, {"x": 1}), '{"x": 1}', 1.0),
    )
    for task, output, expected in cases:
        outcome = router.verify(task, output)
        assert outcome.score == expected
        assert 0.0 <= outcome.score <= 1.0
    assert router.reward(cases[0][0], AgentAction.noop()) == 0.0


class _MockJudgeProvider(ModelProvider):
    def generate(self, request):
        assert request.metadata["role"] == "verifier"
        return ModelResponse(
            provider="mock",
            model="judge",
            text='{"score": 0.75, "is_correct": true, "analysis": "close"}',
        )


def test_llm_judge_uses_provider_neutral_interface() -> None:
    judge = LLMJudgeVerifier(_MockJudgeProvider(), "mock-judge", include_images=False)
    task = _task(
        AnswerType.FREE_FORM,
        "left",
        metadata={"verification_method": "llm_judge"},
    )
    outcome = VerifierRouter(llm_judge=judge).verify(task, "slightly left")
    assert outcome.score == 0.75
    assert outcome.is_correct is True
