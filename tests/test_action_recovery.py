"""Behavioral coverage of output truncation versus environment-step exhaustion."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image
from test_output_budget_recovery import Responses
from test_stage5_agent_rollout import _agent

from spatialcraft.agent.decision import BUDGET_POLICY, parse_complete_action
from spatialcraft.evaluation.efficiency import efficiency
from spatialcraft.experiments.accumulation import KnowledgeState, ProtocolPipeline
from spatialcraft.experiments.journal import RunJournal
from spatialcraft.experiments.rollout import JournaledRollout
from spatialcraft.experiments.settings import ExperimentSettings
from spatialcraft.knowledge.experience import ExperienceBank
from spatialcraft.knowledge.skill import SeedCatalog
from spatialcraft.models import (
    MessageRole,
    ModelResponse,
    ModelResponseError,
    ResponseToolCall,
    TokenUsage,
)
from spatialcraft.schemas import (
    ActionType,
    TaskSample,
    TrajectoryStatus,
)


def text(value, *, truncated=False):
    return ModelResponse(
        provider="test",
        model="test",
        text=value,
        finish_reason="length" if truncated else "stop",
        usage=TokenUsage(input_tokens=20, output_tokens=4096 if truncated else 5),
    )


def tool(image, *, name="detect", truncated=False):
    return ModelResponse(
        provider="test",
        model="test",
        finish_reason="length" if truncated else "stop",
        tool_calls=(
            ResponseToolCall(
                name=name, arguments={"image_uri": str(image), "queries": ["cup"]}
            ),
        ),
    )


def native(image):
    return (
        "<tool_call><function=detect><parameter=image_uri>"
        + str(image)
        + "</parameter>"
        '<parameter=queries>["cup"]</parameter></function></tool_call>'
    )


def setup(tmp_path, responses, *, max_steps=3):
    image = tmp_path / "red.png"
    Image.new("RGB", (32, 32), "red").save(image)
    agent, _ = _agent(tmp_path)
    loop = agent.execution_loop
    loop.composer.model = replace(
        loop.composer.model,
        generation=replace(loop.composer.model.generation, max_output_tokens=4096),
    )
    loop.config = replace(loop.config, max_steps=max_steps)
    source = Responses(*responses(image))
    loop.provider = source
    from spatialcraft.schemas import ImageInput

    task = TaskSample(
        task_id="budget",
        dataset="test",
        question="Is the target visible?",
        images=(ImageInput(uri=str(image)),),
        reference_answer="yes",
    )
    runner = JournaledRollout(
        loop, RunJournal(tmp_path / "journal", {"policy": BUDGET_POLICY})
    )
    state = loop.state_builder.initial(task.without_reference_answer())
    kwargs = {
        "prefix": "rollout",
        "initial_state": state,
        "rollout_index": 0,
        "random_seed": 42,
    }
    return runner, source, task, kwargs


def run_again(runner, source, task, kwargs):
    row = runner.run(task, **kwargs)
    calls = len(source.calls)
    assert runner.run(task, **kwargs).to_dict() == row.to_dict()
    assert len(source.calls) == calls
    return row


def test_recovery_keeps_tools_and_same_state_then_continues(tmp_path):
    runner, source, task, kwargs = setup(
        tmp_path,
        lambda im: [
            text("unfinished reasoning", truncated=True),
            tool(im),
            text("yes"),
        ],
    )
    row = run_again(runner, source, task, kwargs)
    assert row.status is TrajectoryStatus.COMPLETED and row.reward == 1
    assert [t.action.action_type for t in row.transitions] == [
        ActionType.TOOL,
        ActionType.FINAL,
    ]
    original, recovery, after_tool = source.calls
    assert (
        original.settings.max_output_tokens
        == after_tool.settings.max_output_tokens
        == 4096
    )
    assert (
        recovery.settings.max_output_tokens == 1024
        and recovery.settings.temperature == 0
    )
    assert (
        recovery.tools == original.tools and recovery.messages[:-1] == original.messages
    )
    assert recovery.metadata["state_id"] == original.metadata["state_id"]
    assert recovery.metadata["step_index"] == original.metadata["step_index"] == 0
    assert after_tool.metadata["step_index"] == 1
    assert any(m.role is MessageRole.TOOL for m in after_tool.messages)
    assert not any("unfinished reasoning" in m.text_content for m in recovery.messages)
    assert not any("Reference answer:" in m.text_content for m in recovery.messages)
    assert row.metadata["call_counts"] == {
        "normal_llm_calls": 2,
        "recovery_calls": 1,
        "forced_final_answers": 0,
        "token_truncations": 1,
        "tool_calls": 1,
    }
    assert (
        row.transitions[0].metadata["model_request"]["settings"]["max_output_tokens"]
        == 1024
    )
    metrics = efficiency((row,))
    assert metrics.mean_environment_steps == 1 and metrics.mean_recovery_calls == 1
    assert metrics.mean_normal_llm_calls == 2 and metrics.mean_token_truncations == 1


@pytest.mark.parametrize("encoding", ["native", "json", "structured"])
def test_complete_tool_inside_truncated_output_is_executed_without_recovery(
    tmp_path, encoding
):
    def outputs(image):
        if encoding == "native":
            first = text(
                native(image) + "\n<tool_call><function=unfinished", truncated=True
            )
        elif encoding == "json":
            first = text(
                json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "detect",
                                "arguments": {
                                    "image_uri": str(image),
                                    "queries": ["cup"],
                                },
                            }
                        ]
                    }
                )
                + "trailing truncated draft",
                truncated=True,
            )
        else:
            first = tool(image, truncated=True)
        return [first, text("yes")]

    runner, source, task, kwargs = setup(tmp_path, outputs)
    row = run_again(runner, source, task, kwargs)
    assert row.total_tool_calls == 1 and row.status is TrajectoryStatus.COMPLETED
    assert len(source.calls) == 2 and row.metadata["call_counts"]["recovery_calls"] == 0


@pytest.mark.parametrize(
    "answer", ["Final Answer: yes\nunfinished later text", '{"final_answer":"yes"}']
)
def test_complete_truncated_final_needs_no_recovery(tmp_path, answer):
    runner, source, task, kwargs = setup(
        tmp_path, lambda im: [text(answer, truncated=True)]
    )
    row = run_again(runner, source, task, kwargs)
    assert row.status is TrajectoryStatus.COMPLETED and row.reward == 1
    assert len(source.calls) == 1 and row.total_tool_calls == 0


@pytest.mark.parametrize(
    "draft",
    [
        "Final Answer: ye",
        "<tool_call><function=detect>",
        "<think>Final Answer: yes\n",
        '{"tool_calls":[',
        "long unfinished reasoning",
    ],
)
def test_incomplete_action_uses_one_recovery_and_fails_if_still_invalid(
    tmp_path, draft
):
    runner, source, task, kwargs = setup(
        tmp_path, lambda im: [text(draft, truncated=True), text(draft, truncated=True)]
    )
    row = run_again(runner, source, task, kwargs)
    assert (
        row.status is TrajectoryStatus.FAILED
        and row.reward == 0
        and not row.transitions
    )
    assert row.metadata["failure_kind"] == "action_recovery_failed"
    assert (
        len(source.calls) == 2
        and row.metadata["call_counts"]["forced_final_answers"] == 0
    )


@pytest.mark.parametrize(
    "bad_recovery",
    [
        text("<tool_call><function=detect>", truncated=True),
        text(""),
        ModelResponseError("Malformed tool response"),
        ModelResponse(
            provider="test",
            model="test",
            tool_calls=(ResponseToolCall(name="detect", arguments={}),),
        ),
        ModelResponse(
            provider="test",
            model="test",
            tool_calls=(ResponseToolCall(name="unknown", arguments={}),),
        ),
    ],
)
def test_recovery_rejects_invalid_actions_without_crashing_pipeline(
    tmp_path, bad_recovery
):
    runner, source, task, kwargs = setup(
        tmp_path, lambda im: [text("partial", truncated=True), bad_recovery]
    )
    row = run_again(runner, source, task, kwargs)
    assert row.status is TrajectoryStatus.FAILED and row.total_tool_calls == 0
    assert row.metadata["failure_kind"] == "action_recovery_failed"
    assert len(source.calls) == 2


def test_recovery_can_salvage_complete_action_even_at_its_own_cap(tmp_path):
    runner, source, task, kwargs = setup(
        tmp_path,
        lambda im: [
            text("partial", truncated=True),
            text(native(im), truncated=True),
            text("yes"),
        ],
    )
    row = run_again(runner, source, task, kwargs)
    assert row.status is TrajectoryStatus.COMPLETED and row.total_tool_calls == 1
    assert row.metadata["call_counts"]["token_truncations"] == 2


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_step_limit_uses_all_observations_and_one_final_call(tmp_path, answer):
    runner, source, task, kwargs = setup(
        tmp_path, lambda im: [tool(im), tool(im), text(answer)], max_steps=2
    )
    row = run_again(runner, source, task, kwargs)
    assert row.status is TrajectoryStatus.COMPLETED and row.reward == float(
        answer == "yes"
    )
    assert row.total_tool_calls == row.metadata["environment_steps"] == 2
    assert len(row.transitions) == 3 and row.transitions[-1].step_index == 2
    forced = source.calls[-1]
    assert (
        forced.tools == ()
        and forced.tool_choice is None
        and forced.settings.max_output_tokens == 512
    )
    assert len([m for m in forced.messages if m.role is MessageRole.TOOL]) == 2
    assert row.metadata["call_counts"]["forced_final_answers"] == 1
    assert row.transitions[-1].metadata["consumes_environment_step"] is False


@pytest.mark.parametrize("last_kind", ["tool", "truncated", "empty"])
def test_forced_final_never_executes_more_tools_or_recovers(tmp_path, last_kind):
    def outputs(image):
        return [
            tool(image),
            tool(image)
            if last_kind == "tool"
            else text(
                "unfinished" if last_kind == "truncated" else "",
                truncated=last_kind == "truncated",
            ),
        ]

    runner, source, task, kwargs = setup(tmp_path, outputs, max_steps=1)
    row = run_again(runner, source, task, kwargs)
    assert row.status is TrajectoryStatus.FAILED and row.total_tool_calls == 1
    assert row.metadata["failure_kind"] == "forced_final_failed"
    assert len(source.calls) == 2 and row.metadata["call_counts"]["recovery_calls"] == 0


def test_recovery_of_last_tool_step_precedes_forced_final(tmp_path):
    runner, source, task, kwargs = setup(
        tmp_path,
        lambda im: [text("partial", truncated=True), tool(im), text("yes")],
        max_steps=1,
    )
    row = run_again(runner, source, task, kwargs)
    assert row.total_tool_calls == 1 and len(source.calls) == 3
    assert [e["kind"] for e in row.metadata["generation_events"]] == [
        "normal",
        "recovery",
        "forced_final",
    ]


def test_resume_after_recovered_tool_does_not_repeat_recovery_or_tool(tmp_path):
    runner, source, task, kwargs = setup(
        tmp_path,
        lambda im: [
            text("partial", truncated=True),
            tool(im),
            RuntimeError("transport outage"),
            text("yes"),
        ],
    )
    with pytest.raises(RuntimeError, match="transport outage"):
        runner.run(task, **kwargs)
    committed = {p: p.read_bytes() for p in (tmp_path / "journal").rglob("result.json")}
    row = run_again(runner, source, task, kwargs)
    assert row.total_tool_calls == 1 and len(source.calls) == 4
    assert row.metadata["call_counts"]["recovery_calls"] == 1
    assert all(p.read_bytes() == value for p, value in committed.items())


def test_model_failure_is_training_zero_but_infrastructure_failure_is_rejected(
    tmp_path,
):
    runner, _source, task, kwargs = setup(
        tmp_path,
        lambda im: [text("partial", truncated=True), text("partial", truncated=True)],
    )
    row = runner.run(task, **kwargs)
    frozen = KnowledgeState(ExperienceBank().freeze(), SeedCatalog.pool().freeze())
    row = replace(row, knowledge_snapshot_id=frozen.snapshot_id)
    pipeline = ProtocolPipeline(
        ExperimentSettings(),
        runner.journal,
        prepare_experiences=lambda *a: (),
        rollout=lambda *a: row,
        update_experiences=lambda *a: {},
        evolve_skills=lambda *a: {},
    )
    pipeline._validate(row, task, frozen, 0, 42)
    with pytest.raises(ValueError):
        pipeline._validate(replace(row, metadata={}), task, frozen, 0, 42)


def test_multiple_tool_actions_are_not_executed_as_one_step(tmp_path):
    def outputs(image):
        first = tool(image)
        return [replace(first, tool_calls=(*first.tool_calls, *tool(image).tool_calls))]

    runner, source, task, kwargs = setup(tmp_path, outputs)
    row = run_again(runner, source, task, kwargs)
    assert row.status is TrajectoryStatus.FAILED and row.total_tool_calls == 0
    assert len(source.calls) == 1


def test_generic_loop_uses_same_recovery_policy(tmp_path):
    runner, _source, task, _kwargs = setup(
        tmp_path,
        lambda im: [text("partial", truncated=True), tool(im), text("yes")],
        max_steps=1,
    )
    row = runner.loop.run(task, random_seed=42)
    assert row.status is TrajectoryStatus.COMPLETED and row.total_tool_calls == 1
    assert [e["kind"] for e in row.metadata["generation_events"]] == [
        "normal",
        "recovery",
        "forced_final",
    ]


def test_closed_thinking_prefix_is_preserved_for_salvaged_action(tmp_path):
    runner, _source, task, kwargs = setup(tmp_path, lambda im: [])
    request = runner.loop.composer.compose(
        task.without_reference_answer(), kwargs["initial_state"]
    )
    request = replace(
        request,
        metadata={
            **request.metadata,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    result = text(
        "Reasoning here.\n</think>\n\n" + native(Path(task.images[0].uri)),
        truncated=True,
    )
    action, parsed = parse_complete_action(runner.loop.action_parser, result, request)
    assert action.action_type is ActionType.TOOL
    assert parsed.raw["sampled_thinking_prefix"] == "Reasoning here.\n</think>\n\n"


def test_final_marker_inside_partial_tool_arguments_is_not_a_final(tmp_path):
    draft = "<tool_call><function=detect><parameter=queries>\nFinal Answer: yes\n"
    runner, source, task, kwargs = setup(
        tmp_path, lambda im: [text(draft, truncated=True), tool(im), text("yes")]
    )
    row = run_again(runner, source, task, kwargs)
    assert (
        row.total_tool_calls == 1 and row.metadata["call_counts"]["recovery_calls"] == 1
    )


def test_recovery_prose_is_not_mistaken_for_final_answer(tmp_path):
    runner, source, task, kwargs = setup(
        tmp_path,
        lambda im: [
            text("partial", truncated=True),
            text("I should inspect the image next."),
        ],
    )
    row = run_again(runner, source, task, kwargs)
    assert row.status is TrajectoryStatus.FAILED and row.final_answer is None


def test_recovery_may_finish_when_it_explicitly_has_an_answer(tmp_path):
    runner, source, task, kwargs = setup(
        tmp_path,
        lambda im: [text("partial", truncated=True), text("Final Answer: yes")],
    )
    row = run_again(runner, source, task, kwargs)
    assert row.status is TrajectoryStatus.COMPLETED and row.reward == 1
    assert row.metadata["call_counts"]["forced_final_answers"] == 0


def test_runtime_binds_new_trajectory_policy_and_refuses_mixed_resume(tmp_path):
    from spatialcraft.experiments.runtime import ExperimentRuntime

    runtime = ExperimentRuntime(
        Path(__file__).resolve().parents[1], tmp_path, ExperimentSettings(), {}
    )
    journal = runtime.dataset("omni3d").journal
    policy = journal.binding["trajectory_control"]
    assert policy == {
        "policy": BUDGET_POLICY,
        "max_tool_steps": 50,
        "actions_per_step": 1,
        "max_recovery_calls_per_step": 1,
        "recovery_max_tokens": 1024,
        "forced_final_max_tokens": 512,
    }
    with pytest.raises(ValueError, match="binding changed"):
        RunJournal(
            journal.root,
            {
                **journal.binding,
                "trajectory_control": {"policy": "force_completion_v1"},
            },
        )
