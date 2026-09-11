"""Protocol metrics use real decision audit paths and preserve missing evidence."""

import json
from dataclasses import replace

import pytest
from test_action_recovery import setup, text, tool
from test_runtime_v2 import ScriptedModel, runtime, tasks

from spatialcraft.evaluation.protocol_metrics_v2 import protocol_metrics
from spatialcraft.experiments.accumulation import KnowledgeState
from spatialcraft.knowledge.experience import ExperienceBank
from spatialcraft.knowledge.skill import SeedCatalog
from spatialcraft.models import ModelResponseError
from spatialcraft.schemas import (
    ActiveSkillRef,
    ExperienceItem,
    ExperienceStatus,
    RetrievedExperienceRef,
    TaskSplit,
)


def knowledge(*experiences):
    return KnowledgeState(
        ExperienceBank(experiences).freeze(), SeedCatalog.pool().freeze()
    )


def test_recovery_repetition_and_parse_denominators_from_real_decide(tmp_path):
    def outputs(image):
        first = tool(image)
        reordered = replace(
            first,
            tool_calls=(
                replace(
                    first.tool_calls[0],
                    arguments={"queries": ["cup"], "image_uri": str(image)},
                ),
            ),
        )
        return [text("unfinished", truncated=True), first, reordered, text("yes")]

    runner, _, task, kwargs = setup(tmp_path, outputs)
    row = runner.run(task, **kwargs)
    report = protocol_metrics((row,), knowledge())
    rates = report["rates"]
    assert rates["token_truncation"]["rate"] == 1 / 4
    assert rates["recovery_call"]["rate"] == 1 / 4
    assert rates["recovery_success"]["rate"] == 1
    assert rates["action_parse_failure"]["rate"] == 1 / 4
    assert rates["action_structure_or_protocol_error"]["rate"] == 1 / 4
    assert rates["parser_tool_argument_error"]["rate"] == 0
    assert rates["repeated_tool_call"]["numerator"] == 1
    assert rates["repeated_tool_call"]["denominator"] == 2
    doubled = protocol_metrics(
        (row, replace(row, trajectory_id="another")), knowledge()
    )
    assert doubled["rates"]["repeated_tool_call"]["numerator"] == 2
    assert doubled["rates"]["repeated_tool_call"]["denominator"] == 4


@pytest.mark.parametrize("kind", ["arguments", "structure", "provider"])
def test_parser_argument_error_is_distinct_from_format_and_provider_errors(
    tmp_path, kind
):
    def outputs(image):
        if kind == "arguments":
            response = tool(image)
            return [
                replace(
                    response,
                    tool_calls=(
                        replace(
                            response.tool_calls[0],
                            arguments={"image_uri": str(image), "queries": 4},
                        ),
                    ),
                )
            ]
        if kind == "provider":
            return [ModelResponseError("provider returned an invalid response")]
        return [text("<tool_call>")]

    runner, _, task, kwargs = setup(tmp_path, outputs)
    row = runner.run(task, **kwargs)
    assert not row.transitions
    event = row.metadata["generation_events"][0]
    assert event["parse_attempted"] is (kind != "provider")
    assert (
        event["parse_error_type"]
        == {
            "arguments": "argument_validation",
            "structure": "action_parse",
            "provider": "provider_response",
        }[kind]
    )
    rates = protocol_metrics((row,), knowledge())["rates"]
    assert rates["action_parse_failure"]["rate"] == (0 if kind == "provider" else 1)
    assert rates["parser_tool_argument_error"]["rate"] == (
        1 if kind == "arguments" else 0
    )
    assert rates["action_structure_or_protocol_error"]["rate"] == (
        1 if kind == "structure" else 0
    )
    assert rates["repeated_tool_call"]["rate"] is None


def test_old_missing_events_and_parse_flags_remain_unknown(tmp_path):
    runner, _, task, kwargs = setup(tmp_path, lambda image: [text("yes")])
    row = runner.run(task, **kwargs)
    old = replace(
        row, metadata={}, transitions=(replace(row.transitions[0], metadata={}),)
    )
    report = protocol_metrics((old,), knowledge())
    assert report["rates"]["token_truncation"]["rate"] is None
    assert report["rates"]["token_truncation"]["unknown_trajectories"] == 1
    assert (
        report["injection"]["skill"]["transition_coverage"]["unknown_observations"] == 1
    )
    assert report["injection"]["skill"]["block_lengths"]["known_total_tokens"] is None
    event = {
        "kind": "normal",
        "token_truncated": False,
        "parse_error": "old ambiguous error",
    }
    legacy_event = replace(old, metadata={"generation_events": [event]})
    report = protocol_metrics((legacy_event,), knowledge())
    assert report["rates"]["token_truncation"]["rate"] == 0
    assert report["rates"]["action_parse_failure"]["rate"] is None
    assert report["rates"]["parser_tool_argument_error"]["rate"] is None


def test_exact_saved_injection_lengths_and_active_only_storage(tmp_path):
    experience = ExperienceItem(condition="观察左侧", action="确认观察者坐标。")
    bank = knowledge(experience)
    skill = bank.skills.active()[0]
    runner, _, task, kwargs = setup(tmp_path, lambda image: [text("yes")])
    kwargs["initial_state"] = replace(
        kwargs["initial_state"],
        retrieved_experiences=(
            RetrievedExperienceRef(
                experience_id=experience.experience_id,
                version=1,
                retrieval_score=0.9,
                original_text=experience.prompt_text,
                contextualized_text="先定位当前观察者。",
            ),
        ),
        active_skill=ActiveSkillRef(
            skill_id=skill.skill_id, version=skill.version, activated_at_step=0
        ),
        metadata={"active_skill_prompt": skill.format_for_prompt()},
    )
    row = runner.run(task, **kwargs)
    report = protocol_metrics((row,), bank)
    block = report["injection"]["experience"]
    assert block["transition_coverage"]["rate"] == 1
    assert block["trajectory_coverage"]["rate"] == 1
    messages = row.transitions[0].metadata["model_request"]["messages"]
    actual = next(
        m["content"][0]["text"]
        for m in messages
        if m["content"][0].get("text", "").startswith("Retrieved experience")
    )
    assert block["block_lengths"]["known_total_characters"] == len(actual)
    assert block["block_lengths"]["known_total_utf8_bytes"] == len(
        actual.encode("utf-8")
    )
    assert block["block_lengths"]["known_total_tokens"] is None
    assert report["injection"]["skill"]["transition_coverage"]["rate"] == 1
    assert report["storage"]["experience"]["active_items"] == 1
    archived = ExperienceBank(
        (replace(experience, status=ExperienceStatus.ARCHIVED),)
    ).freeze()
    assert (
        protocol_metrics((row,), KnowledgeState(archived, bank.skills))["storage"][
            "experience"
        ]["active_items"]
        == 0
    )


def test_v2_deployment_exports_metrics_and_uses_cached_executor_tokenizer(tmp_path):
    model = ScriptedModel()
    pipeline = runtime(tmp_path, model).dataset("fixture")
    frozen = pipeline.accumulate(tasks(tmp_path))
    test = replace(tasks(tmp_path)[0], task_id="heldout", split=TaskSplit.TEST)
    result = pipeline.deploy((test,), frozen)
    report = json.loads(
        (pipeline.journal.root / result["protocol_metrics_path"]).read_text()
    )
    assert report["trajectory_count"] == 1
    assert report["rates"]["action_parse_failure"]["rate"] == 0
    assert report["tokenizer"]["identity"]["model_alias"] == "qwen3.5-9b"
    assert (
        report["storage"]["skill"]["rendered_contents"]["token_count_status"]
        == "available"
    )
    assert report["storage"]["skill"]["rendered_contents"]["known_total_tokens"] > 0
    requests = len(model.requests)
    assert pipeline.deploy((test,), frozen) == result
    assert len(model.requests) == requests
