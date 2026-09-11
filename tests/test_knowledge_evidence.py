"""Knowledge evidence omits replay audit without dropping spatial observations."""

import json
from dataclasses import replace

from spatialcraft.agent import StateBuilder
from spatialcraft.knowledge.evidence import (
    render_action,
    render_state_evidence,
    render_tool_result,
    render_transition,
)
from spatialcraft.schemas import (
    AgentAction,
    ArtifactRef,
    ArtifactType,
    CoordinateFrame,
    ToolCall,
    ToolResult,
    ToolStatus,
    Transition,
)


def observed_row(row, *, raw=False):
    call = ToolCall(
        tool_name="geometry", arguments={"points": [[1, 2, 3]], "frame_id": "world"}
    )
    action = AgentAction.tool(call)
    result = ToolResult(
        tool_call_id=call.call_id,
        tool_name="geometry",
        status=ToolStatus.SUCCEEDED,
        text="Object A is 2.5 estimated meters to the left.",
        structured_output={
            "distance": 2.5,
            "valid": True,
            "scale_status": "estimated_metric",
            "unit": "meter",
        },
        artifacts=(
            ArtifactRef(
                artifact_type=ArtifactType.JSON,
                artifact_id="random-artifact-identity",
                uri="artifact://random-run/object.json",
                frame_id="world",
            ),
        ),
        coordinate_frames=(
            CoordinateFrame(
                frame_id="world",
                unit="meter",
                convention="right-handed",
                metadata={"scale_status": "estimated_metric"},
            ),
        ),
    )
    before = row.transitions[0].state_before
    after = StateBuilder().after_tools(before, action, (result,))
    steps = (
        Transition(
            step_index=0, state_before=before, action=action, tool_results=(result,)
        ),
        Transition(
            step_index=1, state_before=after, action=AgentAction.final("A"), done=True
        ),
    )
    if raw:
        audit = {
            "sentinel": "RAW_AUDIT_MUST_NOT_BE_PROMPT",
            "token_ids": list(range(1200)),
            "request": {"input_ids": list(range(600))},
        }
        steps = tuple(
            replace(
                step,
                action=replace(
                    step.action, raw_response=audit, metadata={"request": audit}
                ),
            )
            for step in steps
        )
    return replace(row, transitions=steps)


def test_semantic_renderers_preserve_geometry_and_leave_raw_journal_unchanged():
    from test_memory_baselines_v2 import Rollout, task

    from spatialcraft.experiments.accumulation import KnowledgeState
    from spatialcraft.knowledge.experience import ExperienceBank
    from spatialcraft.knowledge.skill import SkillPool
    from spatialcraft.schemas import TaskSplit

    state = KnowledgeState(ExperienceBank().freeze(), SkillPool().freeze())
    row = observed_row(
        Rollout()(task("train", TaskSplit.TRAIN), state, (), 0, 1, "unused", False),
        raw=True,
    )
    original = row.to_dict()
    action = render_action(row.transitions[0].action)
    result = render_tool_result(row.transitions[0].tool_results[0])
    evidence = render_state_evidence(row.transitions[1].state_before)
    combined = json.dumps(
        [action, result, evidence, render_transition(row.transitions[0])]
    )
    assert "RAW_AUDIT_MUST_NOT_BE_PROMPT" not in combined
    assert "token_ids" not in combined and "created_at" not in combined
    assert action["tool_calls"][0]["arguments"]["points"] == [[1, 2, 3]]
    assert result["structured_output"]["distance"] == 2.5
    assert result["coordinate_frames"][0]["unit"] == "meter"
    assert result["artifacts"][0]["frame_id"] == "world"
    assert (
        evidence["tool_observations"][0]["observation"]["structured_output"][
            "scale_status"
        ]
        == "estimated_metric"
    )
    assert "RAW_AUDIT_MUST_NOT_BE_PROMPT" in json.dumps(row.to_dict())
    assert row.to_dict() == original
