import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from spatialcraft.experiments.accumulation import KnowledgeState, ProtocolPipeline
from spatialcraft.experiments.evolution_queue import EvolutionQueue
from spatialcraft.experiments.journal import RunJournal
from spatialcraft.experiments.runtime import ExperimentRuntime
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.knowledge.experience import ExperienceBank, HashingEmbedder
from spatialcraft.models import (
    ContentKind,
    MessageRole,
    ModelProvider,
    ModelResponse,
    ResponseToolCall,
    SequenceScore,
)
from spatialcraft.schemas import (
    ActiveSkillRef,
    AgentAction,
    AnswerType,
    ExperienceItem,
    ExperienceOperationType,
    ExperienceUpdate,
    ImageInput,
    SpatialState,
    TaskSample,
    TaskSplit,
    Trajectory,
    TrajectoryStatus,
    Transition,
)
from spatialcraft.storage.atomic_io import read_json, sha256_file
from spatialcraft.tools import create_mock_tool_registry


def completed(task, frozen, index, seed):
    parent = frozen.skills.active()[0]
    active = ActiveSkillRef(
        skill_id=parent.skill_id, version=parent.version, activated_at_step=0
    )
    state = SpatialState(task_id=task.task_id, step_index=0, active_skill=active)
    action = AgentAction.final("A")
    now = datetime.now(timezone.utc)
    return Trajectory(
        task=task,
        rollout_index=index,
        random_seed=seed,
        executor_model="test",
        knowledge_snapshot_id=frozen.snapshot_id,
        status=TrajectoryStatus.COMPLETED,
        transitions=(
            Transition(
                step_index=0,
                state_before=state,
                state_after=state.next_step(),
                action=action,
                done=True,
                reward=1.0,
                active_skill=active,
            ),
        ),
        reward=1.0,
        started_at=now,
        finished_at=now,
    )


def test_task_barrier_batch_evolution_and_resume(tmp_path):
    tasks = tuple(
        TaskSample(
            task_id=f"t{i}",
            dataset="test",
            question=f"q{i}",
            reference_answer="A",
            split=TaskSplit.TRAIN,
            metadata={"question_type": "test"},
        )
        for i in range(3)
    )
    settings = ExperimentSettings()
    calls = {"retrieve": [], "rollout": [], "experience": [], "skill": []}
    fail = {"enabled": True}

    def retrieve(task, frozen):
        assert task.reference_answer is None
        calls["retrieve"].append(task.task_id)
        return ()

    def rollout(task, frozen, refs, index, seed, prefix, deployment):
        if fail["enabled"] and task.task_id == "t1" and index == 2:
            raise KeyboardInterrupt()
        calls["rollout"].append(
            (
                task.task_id,
                index,
                seed,
                len(frozen.experiences.active()),
                frozen.snapshot_id,
            )
        )
        return completed(task, frozen, index, seed)

    def experience(frozen, rows):
        assert len(rows) == 4 and len({r.knowledge_snapshot_id for r in rows}) == 1
        calls["experience"].append(rows[0].task.task_id)
        bank = ExperienceBank(frozen.experiences.all()).apply(
            ExperienceUpdate(
                operation=ExperienceOperationType.ADD,
                rationale="test",
                proposed_experience=ExperienceItem(
                    condition=rows[0].task.task_id, action="lesson"
                ),
            )
        )
        return {
            "experience_bank": bank.to_dict(),
            "trajectory_ids": [r.trajectory_id for r in rows],
        }

    def evolve(frozen, rows, round_index, state):
        queue = EvolutionQueue(state)
        queue.enqueue(
            rows,
            task_index=round_index,
            active_references={s.reference for s in frozen.skills.active()},
        )
        for entries in queue.take_round().values():
            calls["skill"].append((round_index, len(entries)))
        return {"skill_pool": frozen.skills.to_dict(), "evolution_state": queue.state}

    def build():
        return ProtocolPipeline(
            settings,
            RunJournal(tmp_path, {"test": "barriers"}),
            prepare_experiences=retrieve,
            rollout=rollout,
            update_experiences=experience,
            evolve_skills=evolve,
        )

    with pytest.raises(KeyboardInterrupt):
        build().accumulate(tasks)
    assert calls["experience"] == ["t0"] and not calls["skill"]
    fail["enabled"] = False
    final = build().accumulate(tasks)
    assert calls["retrieve"] == ["t0", "t1", "t2"]
    assert (
        len(calls["rollout"]) == 12 and len({row[2] for row in calls["rollout"]}) == 12
    )
    for task_index in range(3):
        group = [r for r in calls["rollout"] if r[0] == f"t{task_index}"]
        assert {r[3] for r in group} == {task_index}
        assert len({r[4] for r in group}) == 1
    assert calls["experience"] == ["t0", "t1", "t2"]
    assert calls["skill"] == [(1, 6), (2, 6)]
    assert len(final.experiences.active()) == 3
    before = json.dumps(calls, sort_keys=True)
    assert build().accumulate(tasks).snapshot_id == final.snapshot_id
    assert json.dumps(calls, sort_keys=True) == before
    with pytest.raises(ValueError):
        build().accumulate((replace(tasks[0], split=TaskSplit.TEST),))
    with pytest.raises(ValueError, match="overlap"):
        build().deploy((replace(tasks[0], split=TaskSplit.TEST),), final)
    test = replace(tasks[0], task_id="heldout", split=TaskSplit.TEST)
    with pytest.raises(ValueError, match="benchmark"):
        build().deploy((replace(test, dataset="other"),), final)
    with pytest.raises(ValueError, match="final snapshot"):
        build().deploy((test,), KnowledgeState(ExperienceBank().freeze(), final.skills))
    deployed = build().deploy((test,), final)
    count = len(calls["rollout"])
    assert build().deploy((test,), final) == deployed
    assert len(calls["rollout"]) == count


class ScriptedLearningModel(ModelProvider):
    """Offline stand-in; exercises the real runtime/builders, not model quality."""

    def __init__(self):
        self.calls = []
        self.scored = []
        self.visual_summaries = 0
        self.interrupt_next_tool_result = False
        self.interrupt_next_candidate = False

    def generate(self, request):
        response = self._generate(request)
        return replace(
            response,
            raw={"sampled_thinking_prefix": "Offline test reasoning.\n</think>\n\n"},
        )

    def _generate(self, request):
        self.calls.append(request)
        if request.tools:
            has_tool = any(m.role is MessageRole.TOOL for m in request.messages)
            if has_tool and self.interrupt_next_tool_result:
                self.interrupt_next_tool_result = False
                raise KeyboardInterrupt()
            if not has_tool:
                uri = next(
                    p.uri
                    for m in request.messages
                    for p in m.content
                    if p.kind is ContentKind.IMAGE
                )
                return ModelResponse(
                    provider="offline-scripted",
                    model="test",
                    tool_calls=(
                        ResponseToolCall(
                            name="detect",
                            arguments={"image_uri": uri, "queries": ["cup"]},
                        ),
                    ),
                )
            seed = request.settings.seed // 100003
            return ModelResponse(
                provider="offline-scripted",
                model="test",
                text=f"Final Answer: {1 if seed % 2 == 0 else 2}",
            )
        text = request.messages[-1].text_content
        if text.startswith("Decompose"):
            answer = "locate the cup"
        elif text.startswith("Filter then rewrite"):
            answer = "Use current tool evidence, never the remembered answer."
        elif text.startswith("Summarize only"):
            assert any(
                p.kind is ContentKind.IMAGE for m in request.messages for p in m.content
            )
            self.visual_summaries += 1
            answer = "Observed a cup; the tool found its bounding box."
        elif text.startswith("Derive one reusable"):
            answer = "CONDITION: When locating cups\nACTION: Inspect tool evidence before selecting a relation."
        elif text.startswith("Decide whether"):
            answer = '{"terminate":true,"reason":"tool evidence obtained"}'
        elif text.startswith("Diagnose only"):
            answer = json.dumps(
                {
                    "diagnosis": "window only",
                    "initiation": "when a cup relation is queried",
                    "policy": "compare box centers",
                    "termination": "after evidence supports relation",
                }
            )
        elif text.startswith("Generate one reusable"):
            if self.interrupt_next_candidate:
                self.interrupt_next_candidate = False
                raise KeyboardInterrupt()
            index = int(text.rsplit("Candidate index: ", 1)[1])
            answer = json.dumps(
                {
                    "name": f"candidate-{index}",
                    "initiation": "when comparing cup relations",
                    "policy": [f"candidate-{index} evidence rule"],
                    "termination": "after relation evidence",
                }
            )
        else:
            raise AssertionError(text[:120])
        return ModelResponse(provider="offline-scripted", model="test", text=answer)

    def score(self, request, target_text):
        self.scored.append((request, target_text))
        text = "\n".join(m.text_content for m in request.messages)
        delta = 0
        if "candidate-1" in text:
            delta = (
                0.15
                if target_text.endswith("1")
                else -0.15
                if target_text.endswith("2")
                else 0
            )
        return SequenceScore(
            model="test",
            target_text=target_text,
            token_ids=(101,),
            token_logprobs=(-1.0 + delta,),
            prompt_token_count=50,
        )


@pytest.mark.parametrize("thinking", [False, True])
def test_real_runtime_wiring_offline_with_resume_and_read_only_deployment(
    tmp_path, thinking
):
    project = Path(__file__).resolve().parents[1]
    model = ScriptedLearningModel()
    runtime = ExperimentRuntime(
        project,
        tmp_path,
        ExperimentSettings(
            max_steps=4,
            enable_thinking=thinking,
            ppo_thinking_mode="fixed_sampled_prefix" if thinking else "action_only",
        ),
        {"purpose": "offline_test"},
    )
    runtime.local = model
    runtime.embedder = HashingEmbedder(32)
    runtime.tools = create_mock_tool_registry()
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "red").save(image)
    tasks = tuple(
        TaskSample(
            task_id=f"task{i}",
            dataset="robospatial",
            question="Which cup relation?",
            split=TaskSplit.TRAIN,
            answer_type=AnswerType.MULTIPLE_CHOICE,
            choices=("left", "right"),
            reference_answer="A",
            images=(
                ImageInput(
                    uri=str(image), sha256=sha256_file(image), media_type="image/png"
                ),
            ),
            metadata={"question_type": "context", "mask_uri": "DO_NOT_SEND_GT"},
        )
        for i in range(2)
    )
    model.interrupt_next_tool_result = True
    with pytest.raises(KeyboardInterrupt):
        runtime.dataset("robospatial").accumulate(tasks)
    model.interrupt_next_candidate = True
    with pytest.raises(KeyboardInterrupt):
        runtime.dataset("robospatial").accumulate(tasks)
    # Consumption is not committed if candidate construction is interrupted.
    pending = read_json(tmp_path / "robospatial/skills/round-00000.json")[
        "pending_counts"
    ]
    assert pending and all(count == 4 for count in pending.values())
    assert not (
        tmp_path / "robospatial/stages/evolution/00001/skill_evolution/result.json"
    ).exists()
    finals = runtime.dataset("robospatial").accumulate(tasks)
    assert model.visual_summaries == 8
    assert model.scored
    for request, _ in model.scored:
        assert (
            sum(bool(m.metadata.get("ppo_skill_slot")) for m in request.messages) == 1
        )
        assert any(
            p.kind is ContentKind.IMAGE for m in request.messages for p in m.content
        )
    for request in model.calls:
        assert "DO_NOT_SEND_GT" not in str(request_to_text(request))
    first_audit = read_json(tmp_path / "robospatial/skills/round-00000.json")
    assert not first_audit["evolution_batches"] and not first_audit["candidates"]
    audit = read_json(tmp_path / "robospatial/skills/round-00001.json")
    assert 1 <= len(audit["evolution_batches"]) <= 2
    assert all(len(b["source_trajectory_ids"]) == 6 for b in audit["evolution_batches"])
    assert len(audit["candidates"]) == 3 * len(audit["aggregates"])
    assert set(
        first_audit["source_trajectory_ids"] + audit["source_trajectory_ids"]
    ) == {
        r
        for i in range(2)
        for r in read_json(tmp_path / f"robospatial/experiences/task-{i:05d}.json")[
            "source_trajectory_ids"
        ]
    }
    if thinking:
        assert all(
            request.metadata["fixed_scoring_prefix"].endswith("</think>\n\n")
            for request, _ in model.scored
        )
    else:
        assert all(
            "fixed_scoring_prefix" not in request.metadata
            for request, _ in model.scored
        )
    assert all(request.settings.max_output_tokens == 4096 for request in model.calls)
    for request in model.calls:
        is_candidate = request.messages[-1].text_content.startswith(
            "Generate one reusable"
        )
        assert request.settings.temperature == (
            0.7 if request.tools or is_candidate else 0.0
        )
    assert all(
        request.metadata["chat_template_kwargs"]["enable_thinking"] is thinking
        for request in model.calls
    )
    assert all(
        g["metadata"]["segment_end"] - g["metadata"]["segment_start"] == 1
        for g in audit["semantic_gradients"]
    )
    before = (len(model.calls), len(model.scored), finals.snapshot_id)
    assert (
        runtime.dataset("robospatial").accumulate(tasks).snapshot_id
        == finals.snapshot_id
    )
    assert (len(model.calls), len(model.scored)) == before[:2]
    test = replace(tasks[0], task_id="heldout", split=TaskSplit.TEST)
    result = runtime.dataset("robospatial").deploy((test,), finals)
    assert [r for r in model.calls if r.tools][-1].settings.temperature == 0.0
    assert result["knowledge_updated"] is False and finals.snapshot_id == before[2]
    assert result["count"] == 1 and result["accuracy"] in (0.0, 1.0)
    assert len(list((tmp_path / "robospatial/checkpoints").glob("task-*.json"))) == 2


def request_to_text(request):
    return "\n".join(m.text_content for m in request.messages)


def test_fifty_environment_steps_eight_step_skill_limit_and_resume(tmp_path):
    from spatialcraft.knowledge.skill import SeedCatalog

    class ToolOnlyModel(ScriptedLearningModel):
        def _generate(self, request):
            self.calls.append(request)
            if request.tools:
                uri = next(
                    p.uri
                    for m in request.messages
                    for p in m.content
                    if p.kind is ContentKind.IMAGE
                )
                return ModelResponse(
                    provider="offline",
                    model="test",
                    tool_calls=(
                        ResponseToolCall(
                            name="detect",
                            arguments={"image_uri": uri, "queries": ["cup"]},
                        ),
                    ),
                )
            if request.metadata.get("rollout_budget", {}).get("call_kind") == "forced_final":
                return ModelResponse(provider="offline", model="test", text="A")
            assert request.messages[-1].text_content.startswith("Decide whether")
            return ModelResponse(
                provider="offline",
                model="test",
                text='{"terminate": false, "reason": "more evidence needed"}',
            )

    runtime = ExperimentRuntime(
        Path(__file__).resolve().parents[1],
        tmp_path,
        ExperimentSettings(),
        {"purpose": "50_step_boundary"},
    )
    runtime.local = ToolOnlyModel()
    runtime.embedder = HashingEmbedder(32)
    runtime.tools = create_mock_tool_registry()
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "red").save(image)
    task = TaskSample(
        task_id="long-task",
        dataset="robospatial",
        question="Locate the cup",
        reference_answer="A",
        split=TaskSplit.TRAIN,
        images=(ImageInput(uri=str(image)),),
    )
    frozen = KnowledgeState(ExperienceBank().freeze(), SeedCatalog.pool().freeze())
    pipeline = runtime.dataset("robospatial")
    row = pipeline.rollout(task, frozen, (), 0, 42, "long_rollout", False)
    assert row.status is TrajectoryStatus.COMPLETED and row.reward == 1
    assert len(row.transitions) == 51 and row.total_tool_calls == 50
    assert row.metadata["call_counts"]["forced_final_answers"] == 1
    assert row.transitions[-1].metadata["consumes_environment_step"] is False
    assert [t.active_skill.age_steps for t in row.transitions[:-1]] == [
        i % 8 for i in range(50)
    ]
    for step in (7, 15, 23, 31, 39, 47):
        transition = row.transitions[step]
        assert transition.state_after.active_skill is None and not transition.done
        assert row.transitions[step + 1].active_skill.activated_at_step == step + 1
    agent_calls = [r for r in runtime.local.calls if r.tools]
    assert len(agent_calls) == 50
    assert all(
        r.settings.max_output_tokens == 4096 and r.settings.temperature == 0.7
        for r in agent_calls
    )
    count = len(runtime.local.calls)
    again = pipeline.rollout(task, frozen, (), 0, 42, "long_rollout", False)
    assert again.to_dict() == row.to_dict() and len(runtime.local.calls) == count
