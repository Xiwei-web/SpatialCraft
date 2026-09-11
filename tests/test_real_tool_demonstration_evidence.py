"""Real geometry chains have portable demonstrations and distinct durable audits."""

import json

import numpy as np

from spatialcraft.agent import StateBuilder
from spatialcraft.experiments.baseline_memory import render_demonstration
from spatialcraft.knowledge.evidence import render_tool_result
from spatialcraft.schemas import (
    AgentAction,
    TaskSample,
    TaskSplit,
    ToolCall,
    Trajectory,
    TrajectoryStatus,
    Transition,
)
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import sha256_file
from spatialcraft.tools import ArtifactStore, ToolExecutor, ToolRegistry
from spatialcraft.tools.builtin import GeometryTool
from spatialcraft.tools.spatial_arrays import read_bundle


def geometry_demonstration(root, run_id):
    registry = ToolRegistry()
    registry.register(GeometryTool())
    store = ArtifactStore(StorageLayout(root), run_id)
    executor = ToolExecutor(registry, store)
    task = TaskSample(
        task_id="projection-task",
        dataset="cpu_geometry_fixture",
        question="Where does the translated point project in the camera image?",
        split=TaskSplit.TRAIN,
        reference_answer="[100, 50]",
    )
    builder = StateBuilder()
    state = builder.initial(task.without_reference_answer())
    matrix = np.eye(4)
    matrix[0, 3] = 1
    transform = ToolCall(
        tool_name="geometry",
        arguments={
            "operation": "transform_points",
            "points": [[0, 0, 2]],
            "matrix": matrix.tolist(),
            "source_frame_id": "source-world",
            "target_frame_id": "transformed-world",
            "length_unit": "meter",
            "scale_status": "estimated_metric",
        },
    )
    moved = executor.execute(transform)
    assert moved.succeeded, moved.error_message
    first_action = AgentAction.tool(transform)
    after_move = builder.after_tools(state, first_action, (moved,))
    project = ToolCall(
        tool_name="geometry",
        arguments={
            "operation": "project_points",
            "points_uri": moved.artifacts[0].uri,
            "intrinsics": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
            "camera_to_world": np.eye(4).tolist(),
        },
    )
    projected = executor.execute(project)
    assert projected.succeeded, projected.error_message
    second_action = AgentAction.tool(project)
    after_project = builder.after_tools(after_move, second_action, (projected,))
    row = Trajectory(
        task=task,
        executor_model="no-model-real-geometry-fixture",
        rollout_index=0,
        knowledge_snapshot_id="no-knowledge-real-geometry-fixture",
        reward=1.0,
        status=TrajectoryStatus.COMPLETED,
        started_at=moved.started_at,
        finished_at=projected.finished_at,
        transitions=(
            Transition(
                step_index=0,
                state_before=state,
                action=first_action,
                tool_results=(moved,),
                state_after=after_move,
            ),
            Transition(
                step_index=1,
                state_before=after_move,
                action=second_action,
                tool_results=(projected,),
                state_after=after_project,
            ),
            Transition(
                step_index=2,
                state_before=after_project,
                action=AgentAction.final("[100, 50]"),
                done=True,
            ),
        ),
    )
    np.testing.assert_allclose(projected.structured_output["points"], [[100, 50]])
    np.testing.assert_allclose(
        read_bundle(str(store.resolve(projected.artifacts[0])))["points"], [[100, 50]]
    )
    return row, store


def test_real_tool_chain_demonstration_is_equal_across_directories_and_invocations(
    tmp_path,
):
    first, first_store = geometry_demonstration(tmp_path / "directory-one", "run-one")
    second, second_store = geometry_demonstration(tmp_path / "directory-two", "run-two")
    before = (first.to_dict(), second.to_dict())
    disk_before = {
        str(path): path.read_bytes()
        for store in (first_store, second_store)
        for path in store.result_manifests_dir.glob("*.json")
    }
    first_result = first.transitions[1].tool_results[0]
    second_result = second.transitions[1].tool_results[0]
    assert (
        first_result.metadata["invocation_id"]
        != second_result.metadata["invocation_id"]
    )
    assert (
        first_result.metadata["resolved_artifact_uris"]
        != second_result.metadata["resolved_artifact_uris"]
    )
    assert first_result.metadata["resolved_artifact_uris"] == {
        first.transitions[0].tool_results[0].artifacts[0].uri: str(
            first_store.resolve(first.transitions[0].tool_results[0].artifacts[0])
        )
    }

    first_text, second_text = render_demonstration(first), render_demonstration(second)
    assert first_text == second_text
    for forbidden in (
        "invocation_id",
        "resolved_artifact_uris",
        str(tmp_path),
        "run-one",
        "run-two",
    ):
        assert forbidden not in first_text
    content = json.loads(first_text)
    first_artifact = content["steps"][0]["tool_results"][0]["artifacts"][0]
    assert (
        content["steps"][1]["action"]["tool_calls"][0]["arguments"]["points_uri"]
        == first_artifact["uri"]
    )
    assert first_artifact["metadata"]["scale_status"] == "estimated_metric"
    assert content["steps"][1]["tool_results"][0]["structured_output"]["points"] == [
        [100.0, 50.0]
    ]
    assert content["final_answer"] == "[100, 50]"

    # Semantic views never alter the original in-memory or durable executor audit.
    assert (first.to_dict(), second.to_dict()) == before
    for row, store in ((first, first_store), (second, second_store)):
        for step in row.transitions[:2]:
            original = step.tool_results[0]
            restored = store.read_result(original.result_id)
            assert restored.to_dict() == original.to_dict()
            assert restored.metadata["invocation_id"]
            assert "resolved_artifact_uris" in restored.metadata
            for artifact in restored.artifacts:
                assert sha256_file(store.resolve(artifact)) == artifact.sha256
    assert {
        str(path): path.read_bytes()
        for store in (first_store, second_store)
        for path in store.result_manifests_dir.glob("*.json")
    } == disk_before


def test_real_tool_evidence_keeps_spatial_metadata_when_executor_audit_is_removed(
    tmp_path,
):
    row, _ = geometry_demonstration(tmp_path, "spatial-metadata")
    source = row.transitions[1].tool_results[0]
    semantic = render_tool_result(source)
    assert semantic["metadata"]["unit"] == "pixel"
    assert semantic["metadata"]["tool_version"] == source.metadata["tool_version"]
    assert "invocation_id" not in semantic["metadata"]
    assert "resolved_artifact_uris" not in semantic["metadata"]
    assert semantic["structured_output"]["scale_status"] == "estimated_metric"
    assert semantic["structured_output"]["source_coordinate_length_unit"] == "meter"
    assert semantic["coordinate_frames"][0]["frame_id"] == "transformed-world"
    assert (
        semantic["coordinate_frames"][0]["metadata"]["scale_status"]
        == "estimated_metric"
    )
    assert semantic["artifacts"][0]["frame_id"] == source.artifacts[0].frame_id
    assert semantic["artifacts"][0]["metadata"]["world_length_unit"] == "meter"
