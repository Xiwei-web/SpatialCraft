"""Coordinate units stay stable across composed Geometry operations (CPU only)."""

import json

import numpy as np
import pytest
import test_spatial_tools_v2 as spatial_fixture
from test_spatial_tools_v2 import call

from spatialcraft.agent.state_builder import StateBuilder
from spatialcraft.schemas import ActionType, AgentAction, SpatialState, ToolCall
from spatialcraft.tools.spatial_arrays import array_bundle, read_bundle


@pytest.fixture
def scene(tmp_path):
    return spatial_fixture.scene.__wrapped__(tmp_path)


def test_one_world_keeps_length_units_across_angle_rotation_and_projection(scene):
    _, _, executor, _, _ = scene
    state = SpatialState(task_id="geometry-contract", step_index=0)
    outputs = {}
    for operation, arguments in (
        ("point_distance", {"first": [1, 0, 0], "second": [0, 1, 0]}),
        ("angle_between_vectors", {"first": [1, 0, 0], "second": [0, 1, 0]}),
        ("rotation_matrix_from_vectors", {"first": [1, 0, 0], "second": [0, 1, 0]}),
        (
            "project_points",
            {
                "points": [[0, 0, 2]],
                "intrinsics": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
                "camera_to_world": np.eye(4).tolist(),
                "target_frame_id": "camera:pixel",
            },
        ),
    ):
        action = ToolCall(
            tool_name="geometry",
            arguments={
                "operation": operation,
                "frame_id": "world",
                "length_unit": "meter",
                "scale_status": "estimated_metric",
                **arguments,
            },
        )
        result = executor.execute(action)
        assert result.succeeded, result.error_message
        outputs[operation] = result
        state = StateBuilder().after_tools(
            state,
            AgentAction(action_type=ActionType.TOOL, tool_calls=(action,)),
            (result,),
        )
    messages = [
        json.loads(message.content)
        for message in state.messages
        if message.role.value == "tool"
    ]
    assert {
        frame["unit"]
        for message in messages
        for frame in message["coordinate_frames"]
        if frame["frame_id"] == "world"
    } == {"meter"}
    angle = outputs["angle_between_vectors"].structured_output
    assert angle["value_unit"] == angle["unit"] == "degree"
    assert angle["coordinate_length_unit"] == "meter" and angle[
        "angle_degrees"
    ] == pytest.approx(90)
    rotation = outputs["rotation_matrix_from_vectors"].structured_output
    assert (
        rotation["value_unit"] == "dimensionless"
        and rotation["coordinate_length_unit"] == "meter"
    )
    projected = outputs["project_points"]
    assert projected.structured_output["points"] == [[50, 50]]
    assert (
        projected.structured_output["frame_id"]
        == projected.structured_output["result_frame_id"]
        == "camera:pixel"
    )
    assert projected.structured_output["source_frame_id"] == "world"
    assert projected.structured_output["source_coordinate_length_unit"] == "meter"
    assert projected.structured_output["coordinate_length_unit"] == "pixel"
    assert (
        projected.artifacts[0].frame_id == state.evidence[-1].frame_id == "camera:pixel"
    )


def test_estimated_metric_survives_mask_transform_project_artifact_backproject(scene):
    image, mask, executor, store, reconstruction = scene
    scaled = call(
        executor,
        "scale",
        image_uri=str(image),
        reconstruction_uri=reconstruction.artifacts[0].uri,
        frame_index=17,
        min_valid_pixels=8,
    )
    calibrated = next(
        artifact
        for artifact in scaled.artifacts
        if artifact.metadata.get("role") == "calibrated_reconstruction"
    )
    masked = call(
        executor,
        "mask",
        operation="masked_points",
        mask_uris=[str(mask)],
        reconstruction_uri=calibrated.uri,
        frame_index=17,
    )
    matrix = np.eye(4)
    matrix[:3, 3] = [1, 2, 3]
    transformed = call(
        executor,
        "geometry",
        operation="transform_points",
        points_uri=masked.artifacts[0].uri,
        matrix=matrix.tolist(),
        target_frame_id="shifted",
    )
    reconstruction_bundle = read_bundle(str(store.resolve(calibrated)))
    projected = call(
        executor,
        "geometry",
        operation="project_points",
        points_uri=transformed.artifacts[0].uri,
        intrinsics=reconstruction_bundle["intrinsics"][0].tolist(),
        camera_to_world=matrix.tolist(),
        pixel_space="source",
        source_to_processed=reconstruction_bundle["source_to_processed"][0].tolist(),
    )
    projection_bundle = read_bundle(str(store.resolve(projected.artifacts[0])))
    assert str(projection_bundle["pixel_space"]) == "source"
    assert str(projection_bundle["intrinsics_space"]) == "processed"
    assert str(projection_bundle["length_unit"]) == "pixel"
    assert str(projection_bundle["world_length_unit"]) == "meter"
    np.testing.assert_allclose(projection_bundle["pixels"][0], [0.5, 0.5])
    recovered = call(
        executor,
        "geometry",
        operation="backproject_points",
        points_uri=projected.artifacts[0].uri,
        depth_values=[6] * 24,
    )
    expected = read_bundle(str(store.resolve(transformed.artifacts[0])))["points"]
    recovered_bundle = read_bundle(str(store.resolve(recovered.artifacts[0])))
    np.testing.assert_allclose(recovered_bundle["points"], expected, atol=1e-10)
    for result in (transformed, projected, recovered):
        bundle = read_bundle(str(store.resolve(result.artifacts[0])))
        assert (
            str(bundle["scale_status"])
            == result.structured_output["scale_status"]
            == "estimated_metric"
        )
        assert result.artifacts[0].metadata["scale_status"] == "estimated_metric"
    assert (
        recovered.structured_output["source_frame_id"]
        == projected.structured_output["frame_id"]
    )
    assert recovered.structured_output["frame_id"] == "shifted"
    assert recovered.structured_output["coordinate_length_unit"] == "meter"


def test_meter_is_not_evidence_of_verified_scale_for_inline_or_old_artifacts(
    scene, tmp_path
):
    _, _, executor, store, _ = scene
    old = tmp_path / "old_points.npz"
    old.write_bytes(
        array_bundle(points=[[0, 0, 2]], frame_id="world", length_unit="meter")
    )
    for source in (
        {"points": [[0, 0, 2]], "frame_id": "world", "length_unit": "meter"},
        {"points_uri": str(old)},
    ):
        result = call(
            executor,
            "geometry",
            operation="transform_points",
            matrix=np.eye(4).tolist(),
            **source,
        )
        assert result.structured_output["scale_status"] == "unverified"
        assert (
            str(read_bundle(str(store.resolve(result.artifacts[0])))["scale_status"])
            == "unverified"
        )


@pytest.mark.parametrize(
    "bad_arguments,expected",
    [
        ({"target_frame_id": "world"}, "distinct frame"),
        ({"pixel_space": "source"}, "source_to_processed"),
        (
            {
                "pixel_space": "source",
                "source_to_processed": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            },
            "invertible",
        ),
    ],
)
def test_projection_rejects_ambiguous_frames_and_missing_or_invalid_mapping(
    scene, bad_arguments, expected
):
    _, _, executor, _, _ = scene
    result = executor.execute(
        ToolCall(
            tool_name="geometry",
            arguments={
                "operation": "project_points",
                "points": [[0, 0, 2]],
                "frame_id": "world",
                "intrinsics": np.eye(3).tolist(),
                "camera_to_world": np.eye(4).tolist(),
                **bad_arguments,
            },
        )
    )
    assert not result.succeeded and result.error_type == "argument_validation"
    assert expected in result.error_message


def test_artifact_provenance_cannot_be_silently_overridden(scene):
    _, _, executor, _, _ = scene
    projected = call(
        executor,
        "geometry",
        operation="project_points",
        points=[[0, 0, 2]],
        frame_id="world",
        length_unit="meter",
        scale_status="estimated_metric",
        intrinsics=np.eye(3).tolist(),
        camera_to_world=np.eye(4).tolist(),
    )
    for overrides in (
        {"scale_status": "known_metric"},
        {"target_frame_id": "other_world"},
        {"pixel_space": "source"},
        {"length_unit": "reconstruction_unit"},
        {"intrinsics": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]},
    ):
        result = executor.execute(
            ToolCall(
                tool_name="geometry",
                arguments={
                    "operation": "backproject_points",
                    "points_uri": projected.artifacts[0].uri,
                    "depth_values": [2],
                    **overrides,
                },
            )
        )
        assert not result.succeeded and result.error_type == "argument_validation"


def test_invalid_projected_pixels_require_explicit_selection_before_backprojection(
    scene,
):
    _, _, executor, _, _ = scene
    projected = call(
        executor,
        "geometry",
        operation="project_points",
        points=[[0, 0, 2], [0, 0, -1]],
        intrinsics=np.eye(3).tolist(),
        camera_to_world=np.eye(4).tolist(),
    )
    result = executor.execute(
        ToolCall(
            tool_name="geometry",
            arguments={
                "operation": "backproject_points",
                "points_uri": projected.artifacts[0].uri,
                "depth_values": [2, 1],
            },
        )
    )
    assert not result.succeeded and "select valid points" in result.error_message


def test_pixel_normalization_uses_separate_result_frame(scene):
    _, _, executor, _, _ = scene
    result = call(
        executor,
        "geometry",
        operation="convert_points",
        points=[[50, 25]],
        width=101,
        height=51,
        source_unit="pixel",
        target_unit="normalized",
        frame_id="image:pixel",
    )
    assert result.structured_output["points"] == [[0.5, 0.5]]
    assert {frame.frame_id: frame.unit for frame in result.coordinate_frames} == {
        "image:pixel": "pixel",
        "image:pixel:normalized": "normalized",
    }
    assert result.artifacts[0].frame_id == result.structured_output["result_frame_id"]
