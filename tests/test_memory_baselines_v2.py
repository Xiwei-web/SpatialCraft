"""Independent baseline execution, method isolation, replay and frozen deployment."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from spatialcraft.experiments.accumulation import KnowledgeState
from spatialcraft.experiments.baseline_memory import (
    MemoryBaselineConfig,
    MemoryBaselineLearner,
    MemoryBaselinePipeline,
    method_description,
)
from spatialcraft.experiments.journal import RunJournal
from spatialcraft.experiments.run_memory_baseline import build_memory_pipeline, main
from spatialcraft.knowledge.experience import ExperienceBank
from spatialcraft.knowledge.skill import SkillPool
from spatialcraft.schemas import (
    ActionType,
    AgentAction,
    ExperienceItem,
    ImageInput,
    SkillEvolutionType,
    SpatialState,
    TaskSample,
    TaskSplit,
    Trajectory,
    TrajectoryStatus,
    Transition,
)
from spatialcraft.schemas._base import utc_now


class Embedder:
    identity = "baseline-semantic-test"

    def __init__(self):
        self.inputs = []

    def embed(self, texts):
        self.inputs.extend(texts)
        return np.asarray([[1.0, 0.0] for _ in texts]).reshape(-1, 2)


class Responses:
    def __init__(self):
        self.calls = []
        self.scope = {}

    def __call__(self, operation, payload, *, media=(), validator=None):
        self.calls.append((operation, payload, tuple(media), dict(self.scope)))
        if operation == "baseline.reflection":
            value = {
                "memories": [
                    {
                        "condition": "When observer axes are ambiguous",
                        "procedure": [
                            "Identify the observer.",
                            "Check the two objects in one frame.",
                        ],
                        "evidence_refs": [payload["trajectory_id"]],
                    }
                ]
            }
        elif operation == "baseline.workflow":
            value = {
                "operations": [
                    {
                        "type": "add",
                        "name": "Resolve observer frame",
                        "initiation": "When a question changes viewpoint",
                        "policy": ["Identify the observer", "Compare in its frame"],
                        "termination": "The relative direction is resolved",
                    }
                ]
            }
        elif operation == "experience.summary":
            value = {
                "summary": "Checked directions.",
                "observed_facts": ["A was answered."],
                "inferred_causes": [],
                "evidence_refs": [payload["trajectory_id"]],
            }
        elif operation == "experience.critique":
            value = {
                "operations": [
                    {
                        "type": "add",
                        "condition": "When observer axes are ambiguous",
                        "action": "Check observer axes first.",
                        "evidence_refs": [payload["summaries"][0]["trajectory_id"]],
                    }
                ]
            }
        elif operation == "experience.merge":
            value = {"decision": "keep_separate", "reason": "Different contexts."}
        elif operation == "retrieval.decomposition":
            value = {"aspects": [{"type": "frame", "query": "Check observer frame"}]}
        elif operation == "retrieval.rewrite":
            value = {
                "items": [
                    {
                        "source_ref": item["reference"],
                        "decision": "keep",
                        "condition": item["condition"],
                        "action": item["action"],
                    }
                    for item in payload["experiences"]
                ]
            }
        else:
            raise AssertionError(operation)
        if validator:
            validator(value)
        return value


def task(identifier, split):
    return TaskSample(
        task_id=identifier,
        dataset="test",
        split=split,
        question="Which object is left?",
        reference_answer="PRIVATE_GT",
        images=(ImageInput(uri="input.png"),),
    )


class Rollout:
    def __init__(self):
        self.calls = []

    def __call__(self, task, knowledge, refs, index, seed, prefix, deployment):
        self.calls.append(
            (
                task.task_id,
                knowledge.snapshot_id,
                tuple(refs),
                deployment,
                len(knowledge.skills.active()),
            )
        )
        state = SpatialState(
            task_id=task.task_id, step_index=0, retrieved_experiences=refs
        )
        return Trajectory(
            task=task,
            rollout_index=index,
            executor_model="fixture-real-callback",
            knowledge_snapshot_id=knowledge.snapshot_id,
            random_seed=seed,
            trajectory_id=f"{task.task_id}-trajectory-{index}",
            reward=1.0 if index % 2 == 0 else 0.0,
            status=TrajectoryStatus.COMPLETED,
            started_at=utc_now(),
            finished_at=utc_now(),
            transitions=(
                Transition(
                    step_index=0,
                    state_before=state,
                    action=AgentAction(action_type=ActionType.FINAL, final_answer="A"),
                    done=True,
                ),
            ),
        )


@pytest.mark.parametrize(
    "method,expected_memories,expected_skills",
    [
        ("rag_demonstrations", 2, 0),
        ("memp_reflection", 4, 0),
        ("memrl_reward", 4, 0),
        ("sma_procedure", 2, 0),
        ("xskill_dual_memory", 1, 1),
        ("skill_pro_sequence", 0, 6),
    ],
)
def test_independent_methods_execute_and_freeze_with_idempotent_replay(
    tmp_path, method, expected_memories, expected_skills
):
    generated, embedder, rollout = Responses(), Embedder(), Rollout()
    evolutions = []

    def evolve(knowledge, rows, index, evolution_state):
        evolutions.append((knowledge, rows, index, evolution_state))
        source = knowledge.skills.active()[0]
        refined = replace(
            source,
            version=source.version + 1,
            evolution_type=SkillEvolutionType.REFINE,
            parent_skill_ref=source.reference,
            updated_at=utc_now(),
        )
        pool = SkillPool(knowledge.skills.all()).refine(refined)
        return {"skill_pool": pool.to_dict(), "evolution_state": {"count": 1}}

    learner = MemoryBaselineLearner(
        MemoryBaselineConfig(method),
        generate=generated,
        embedder=embedder,
        evolve_skills=evolve,
    )
    pipeline = MemoryBaselinePipeline(
        learner=learner,
        journal=RunJournal(tmp_path, {"method": method}),
        rollout=rollout,
    )
    training = (task("train", TaskSplit.TRAIN),)
    state = pipeline.accumulate(training)
    assert len(state.experiences.active()) == expected_memories
    assert len(state.skills.active()) == expected_skills
    initial_calls, initial_generations = len(rollout.calls), len(generated.calls)
    replay = pipeline.accumulate(training)
    assert replay.snapshot_id == state.snapshot_id
    assert len(rollout.calls) == initial_calls == 4
    assert len(generated.calls) == initial_generations
    before = state.to_dict()
    results = pipeline.deploy((task("test", TaskSplit.TEST),), state)
    assert results["accuracy"] == 1.0 and results["count"] == 1
    assert results["baseline"]["faithful_reproduction"] is False
    assert state.to_dict() == before
    assert rollout.calls[-1][3] is True
    if method == "rag_demonstrations":
        assert generated.calls == []
        assert all(not call[2] for call in rollout.calls[:4])
        assert rollout.calls[-1][2]
    elif method == "skill_pro_sequence":
        assert len(evolutions) == 1 and generated.calls == []
        for index in range(4):
            alias = pipeline.journal.read_committed(
                f"tasks/00000/rollouts/{index:02d}/complete"
            )
            source = pipeline.journal.read_committed(
                f"memory_baseline/training/00000/rollouts/{index:02d}/complete"
            )
            assert alias == source == evolutions[0][1][index].to_dict()
        assert not rollout.calls[-1][2]
        assert any(skill.version == 2 for skill in state.skills.active())
    elif method == "xskill_dual_memory":
        names = [call[0] for call in generated.calls]
        assert "baseline.workflow" in names and "experience.critique" in names
        assert not evolutions
    else:
        assert not evolutions
        assert all(name == "baseline.reflection" for name, *_ in generated.calls)
        assert all(
            scope["phase"] == "accumulation_learning"
            for _, _, _, scope in generated.calls
        )


@pytest.mark.parametrize("method", ["memrl_reward", "sma_procedure"])
def test_utility_reranking_uses_later_training_outcomes_without_ground_truth(method):
    poor = ExperienceItem(
        experience_id="E1",
        condition="When comparing axes",
        action="Check the frame.",
        metadata={"baseline_outcomes": {"r1": 0.0, "r2": 0.0}},
    )
    good = ExperienceItem(
        experience_id="E2",
        condition="When comparing observer axes",
        action="Check the observer frame.",
        metadata={"baseline_outcomes": {"r3": 1.0, "r4": 1.0}},
    )
    state = KnowledgeState(ExperienceBank((poor, good)).freeze(), SkillPool().freeze())
    embedder = Embedder()
    learner = MemoryBaselineLearner(
        MemoryBaselineConfig(method, retrieval_top_k=1),
        generate=Responses(),
        embedder=embedder,
    )
    result = learner.prepare(task("test", TaskSplit.TEST), state, deployment=True)
    assert result["experiences"][0]["experience_id"] == "E2"
    assert result["retrieval_audit"]["uses_reference_answer"] is False
    assert all("PRIVATE_GT" not in text for text in embedder.inputs)
    assert state.experiences.get("E2@1").metadata["baseline_outcomes"] == {
        "r3": 1.0,
        "r4": 1.0,
    }


def test_no_oracle_variant_is_silently_enabled():
    assert method_description("memrl_gt")["available"] is False
    with pytest.raises(ValueError, match="unavailable"):
        MemoryBaselineConfig("memrl_gt")


def test_skill_pro_requires_actual_sequence_gate_callback():
    with pytest.raises(ValueError, match="sequence-gate"):
        MemoryBaselineLearner(
            MemoryBaselineConfig("skill_pro_sequence"),
            generate=Responses(),
            embedder=Embedder(),
        )
    with pytest.raises(ValueError, match="sequence-gate"):
        MemoryBaselineLearner(
            MemoryBaselineConfig("skill_pro_sequence"),
            generate=Responses(),
            embedder=Embedder(),
            evolve_skills=lambda *args: {},
            skill_ratio_mode="mean_token",
        )


def test_shared_runtime_adapter_reuses_actual_rollout_and_audit_hooks(tmp_path):
    generator, embedder, rollout = Responses(), Embedder(), Rollout()
    checked = []
    shared = SimpleNamespace(
        generate_knowledge=generator,
        embedder=embedder,
        artifact_resolver=lambda uri: uri,
        evolve_skills=lambda *args: None,
        journal=RunJournal(tmp_path, {"runtime": "fixture"}),
        rollout=rollout,
        artifact_validator=lambda row: checked.append(row.trajectory_id),
    )
    settings = SimpleNamespace(
        is_v2=True, skill_options={}, experience_options={}, seed=9, rollouts_per_task=4
    )
    pipeline = build_memory_pipeline(
        shared, settings, MemoryBaselineConfig("memp_reflection")
    )
    pipeline.accumulate((task("train", TaskSplit.TRAIN),))
    assert pipeline.rollout is shared.rollout
    assert len(rollout.calls) == len(checked) == 4


def test_deployment_rejects_training_id_overlap(tmp_path):
    learner = MemoryBaselineLearner(
        MemoryBaselineConfig("memp_reflection"),
        generate=Responses(),
        embedder=Embedder(),
    )
    pipeline = MemoryBaselinePipeline(
        learner=learner,
        journal=RunJournal(tmp_path, {"test": "overlap"}),
        rollout=Rollout(),
    )
    state = pipeline.accumulate((task("same", TaskSplit.TRAIN),))
    with pytest.raises(ValueError, match="overlaps"):
        pipeline.deploy((task("same", TaskSplit.TEST),), state)


def test_list_methods_is_offline_and_exposes_adaptation_status(capsys):
    main(["--list-methods"])
    output = capsys.readouterr().out
    assert "paper_inspired_spatial_adapter" in output
    assert '"available": false' in output


def test_cli_accepts_new_sat_and_viewspatial_choices(monkeypatch, tmp_path, capsys):
    from spatialcraft.experiments import run_memory_baseline

    seen = []

    # Stop immediately after argument validation, before any model/preparation IO.
    def preflight(*args):
        seen.append(args[-1])
        raise RuntimeError("parsed-supported-v2-datasets")

    monkeypatch.setattr(run_memory_baseline, "preflight", preflight)
    with pytest.raises(RuntimeError, match="parsed-supported"):
        main(
            [
                "--method",
                "memp_reflection",
                "--config",
                str(tmp_path / "config"),
                "--preparation",
                str(tmp_path / "prep"),
                "--output",
                str(tmp_path / "out"),
                "--datasets",
                "sat",
                "viewspatial",
            ]
        )
    assert seen == [["sat", "viewspatial"]]


def test_xskill_rewrite_failure_keeps_actual_retrieval_audit_and_skips_injection():
    from spatialcraft.experiments.knowledge_generator import KnowledgeValidationError

    generated = Responses()

    def generation(operation, payload, **kwargs):
        if operation == "retrieval.rewrite":
            raise KnowledgeValidationError("invalid rewrite after bounded repair")
        return generated(operation, payload, **kwargs)

    memory = ExperienceItem(
        experience_id="E1",
        condition="When comparing observer axes",
        action="Check the observer frame.",
    )
    state = KnowledgeState(ExperienceBank((memory,)).freeze(), SkillPool().freeze())
    learner = MemoryBaselineLearner(
        MemoryBaselineConfig("xskill_dual_memory"),
        generate=generation,
        embedder=Embedder(),
    )
    result = learner.prepare(task("test", TaskSplit.TEST), state, deployment=True)
    assert result["experiences"] == []
    assert result["retrieval_audit"]["retrieved_refs"] == ["E1@1"]
    assert result["retrieval_audit"]["injected_refs"] == []
    assert result["retrieval_audit"]["status"] == "rejected_model_output"


def test_rag_semantic_demonstrations_ignore_raw_token_audit_and_include_observation():
    import json

    from test_knowledge_evidence import observed_row

    from spatialcraft.experiments.baseline_memory import render_demonstration

    learner = MemoryBaselineLearner(
        MemoryBaselineConfig("rag_demonstrations"),
        generate=Responses(),
        embedder=Embedder(),
    )
    initial = learner.initial()
    base = Rollout()(task("train", TaskSplit.TRAIN), initial, (), 0, 42, "test", False)
    clean = observed_row(base)
    raw = observed_row(base, raw=True)
    # Different audit-generated call/transition IDs must not change corpus text.
    assert render_demonstration(clean) == render_demonstration(raw)
    left = learner.update(initial, (clean,), 0)
    right = learner.update(initial, (raw,), 0)
    clean_bank = KnowledgeState.from_dict(left["knowledge"]).experiences
    raw_bank = KnowledgeState.from_dict(right["knowledge"]).experiences
    assert len(clean_bank.active()) == len(raw_bank.active()) == 1
    text = raw_bank.active()[0].prompt_text
    assert text == clean_bank.active()[0].prompt_text
    assert "2.5" in text and "estimated_metric" in text and "geometry" in text
    for excluded in (
        "raw_response",
        "token_ids",
        "RAW_AUDIT",
        "PRIVATE_GT",
        "random-artifact-identity",
        "random-run",
        "created_at",
        "call_id",
    ):
        assert excluded not in text
    assert '"final_answer": "A"' in text
    assert "RAW_AUDIT" in json.dumps(raw.to_dict())
    assert (
        left["memory_construction"]["saved_memories"]
        == right["memory_construction"]["saved_memories"]
        == 1
    )


def test_rag_token_budget_measures_exact_injected_prompt_and_reports_rejections(
    tmp_path,
):
    measured = []

    def count_tokens(text):
        measured.append(text)
        return 1100

    learner = MemoryBaselineLearner(
        MemoryBaselineConfig("rag_demonstrations", demonstration_max_tokens=1024),
        generate=Responses(),
        embedder=Embedder(),
        token_counter=count_tokens,
        tokenizer_id="test-executor-tokenizer",
    )
    pipeline = MemoryBaselinePipeline(
        learner=learner,
        journal=RunJournal(tmp_path, {"test": "budget"}),
        rollout=Rollout(),
    )
    training = (task("train", TaskSplit.TRAIN),)
    frozen = pipeline.accumulate(training)
    result = pipeline.deploy((task("heldout", TaskSplit.TEST),), frozen)
    assert not frozen.experiences.active()
    assert len(measured) == 2 and all(
        text.startswith("Condition:") for text in measured
    )
    stats = result["memory_construction"]
    assert stats["eligible_successful_trajectories"] == 2
    assert stats["budget_rejected_memories"] == 2
    assert stats["saved_memories"] == stats["active_memories"] == 0
    assert stats["memory_budget"]["unit"] == "token"
    assert stats["memory_budget"]["tokenizer"] == "test-executor-tokenizer"
    assert result["retrieval"]["hit_rate"] == 0
    assert result["retrieval"]["requests"] == 1
    # Replaying committed construction never reruns token admission or execution.
    pipeline.accumulate(training)
    assert len(measured) == 2


def test_rag_real_runtime_callback_saves_raw_audited_rollouts_and_retrieves(tmp_path):
    from test_runtime_v2 import ScriptedModel, runtime, tasks

    model = ScriptedModel()
    r = runtime(tmp_path, model)
    shared = r.dataset("fixture")
    shared.metric_token_counter = lambda text: len(text.split())
    shared.metric_tokenizer_id = "scripted-executor-tokenizer"
    pipeline = build_memory_pipeline(
        shared, r.settings, MemoryBaselineConfig("rag_demonstrations")
    )
    training = tasks(tmp_path)[:1]
    frozen = pipeline.accumulate(training)
    heldout = replace(training[0], task_id="heldout", split=TaskSplit.TEST)
    result = pipeline.deploy((heldout,), frozen)
    assert pipeline.rollout is shared.rollout
    assert result["memory_construction"]["eligible_successful_trajectories"] == 4
    assert result["memory_construction"]["saved_memories"] == 4
    assert result["retrieval"]["hit_rate"] == 1
    assert len(frozen.experiences.active()) == 4
    # No Skill slots exist in this baseline: its shared v2 executor uses fallback.
    assert [request.metadata["operation"] for request in model.requests] == [
        "execution.fallback"
    ] * 5
    stored = shared.journal.read_committed(
        "memory_baseline/training/00000/rollouts/00/complete"
    )
    assert stored["transitions"][0]["action"]["raw_response"]
    assert "token_ids" not in frozen.experiences.active()[0].prompt_text


def test_skill_pro_capacity_rejects_less_than_seed_pool():
    with pytest.raises(ValueError, match="six initial"):
        MemoryBaselineConfig("skill_pro_sequence", skill_capacity=3)


def test_skill_pro_capacity_ten_controls_actual_runtime_evolution(tmp_path):
    from test_runtime_v2 import ScriptedModel, runtime, tasks

    from spatialcraft.experiments.run_memory_baseline import effective_baseline_settings
    from spatialcraft.schemas import SkillItem

    r = runtime(tmp_path, ScriptedModel())
    config = MemoryBaselineConfig("skill_pro_sequence", skill_capacity=10)
    r.settings = effective_baseline_settings(r.settings, config)
    shared = r.dataset("fixture")
    pipeline = build_memory_pipeline(shared, r.settings, config)
    assert shared.settings.skill_capacity == 10
    assert shared.learning_builders.skill.config["capacity"] == 10
    assert pipeline.description["config"]["skill_capacity"] == 10
    state = pipeline.learner.initial()
    pool = SkillPool(state.skills.all())
    for i in range(8):
        pool = pool.add(
            SkillItem(
                skill_id=f"extra-{i}",
                name=f"Distinct {i}",
                initiation=f"Condition {i}",
                policy=(f"Procedure {i}",),
                termination=f"Done {i}",
            )
        )
    state = KnowledgeState(state.experiences, pool.freeze())
    rows = tuple(
        Rollout()(tasks(tmp_path)[0], state, (), i, i, "test", False) for i in range(4)
    )
    # The real bound LearningBuildersV2.evolve callback runs diagnosis/maintenance.
    updated = pipeline.learner.update(state, rows, 0)
    final = KnowledgeState.from_dict(updated["knowledge"])
    assert len(final.skills.active()) == 10
    assert (
        len(
            [
                item
                for item in updated["skill_audit"]["maintenance_operations"]
                if item["type"] == "capacity_archive"
            ]
        )
        == 4
    )


def test_skill_pro_rejects_already_constructed_runtime_capacity_mismatch(tmp_path):
    from test_runtime_v2 import ScriptedModel, runtime

    r = runtime(tmp_path, ScriptedModel())
    shared = r.dataset("fixture")
    with pytest.raises(ValueError, match="effective_baseline_settings"):
        build_memory_pipeline(
            shared,
            r.settings,
            MemoryBaselineConfig("skill_pro_sequence", skill_capacity=10),
        )


def test_skill_pro_cli_preflight_resolves_capacity_before_runtime(
    tmp_path, monkeypatch, capsys
):
    import json

    from spatialcraft.experiments import run_memory_baseline
    from spatialcraft.experiments.settings import ExperimentSettings

    settings = ExperimentSettings(
        protocol_version="spatialcraft_v2", embedding_model="text-embedding-3-large"
    )
    monkeypatch.setattr(
        run_memory_baseline,
        "preflight",
        lambda *args: (settings, {}, {}, {"status": "preflight_passed_not_run"}),
    )
    report = tmp_path / "report.json"
    run_memory_baseline.main(
        [
            "--method",
            "skill_pro_sequence",
            "--config",
            str(tmp_path / "config.yaml"),
            "--preparation",
            str(tmp_path / "prep"),
            "--output",
            str(tmp_path / "out"),
            "--skill-capacity",
            "10",
            "--report",
            str(report),
        ]
    )
    value = json.loads(report.read_text())
    assert (
        value["memory_baseline"]["config"]["skill_capacity"]
        == value["effective_settings"]["skill_capacity"]
        == 10
    )
    assert not (tmp_path / "out").exists()
    capsys.readouterr()
