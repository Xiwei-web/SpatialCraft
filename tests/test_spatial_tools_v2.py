"""CPU analytic geometry and artifact integration; no model/GPU calls."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spatialcraft.schemas import ArtifactType, ToolCall
from spatialcraft.storage import StorageLayout
from spatialcraft.tools import ArtifactStore, ToolExecutor, ToolRegistry
from spatialcraft.tools.base import ToolContext
from spatialcraft.tools.builtin import GeometryTool
from spatialcraft.tools.real import (
    DepthAnything3Adapter,
    MaskTool,
    MoGe2Adapter,
    OrientAnythingAdapter,
    PoseTool,
    ReconstructionTool,
    ScaleTool,
    SceneGraphTool,
)
from spatialcraft.tools.spatial_arrays import read_bundle


@pytest.fixture
def scene(tmp_path):
    image = tmp_path / "scene.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (12, 8), "gray").save(image)
    Image.new("L", (12, 8), 255).save(mask)

    def reconstruct(images):
        return {
            "depth": np.full((len(images), 4, 6), 2.0),
            "conf": np.full((len(images), 4, 6), 1.7),  # raw score, not probability
            "intrinsics": np.repeat(
                np.array([[[2, 0, 2.5], [0, 2, 1.5], [0, 0, 1]]]), len(images), axis=0
            ),
            "extrinsics": np.repeat(np.eye(4)[None], len(images), axis=0),
        }

    def scale(uri, resolution):
        return {
            "depth": np.full((8, 12), 6.0),
            "points": np.full((8, 12, 3), 6.0),
            "intrinsics": np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]),
            "mask": np.ones((8, 12), bool),
        }

    registry = ToolRegistry()
    for tool in (
        ReconstructionTool(DepthAnything3Adapter(predictor=reconstruct)),
        ScaleTool(MoGe2Adapter(predictor=scale)),
        GeometryTool(),
        MaskTool(),
        SceneGraphTool(),
        PoseTool(
            OrientAnythingAdapter(
                predictor=lambda image, bbox: {
                    "azimuth_deg": 0,
                    "polar_deg": 0,
                    "rotation_deg": 0,
                    "confidence": 0.9,
                }
            )
        ),
    ):
        registry.register(tool)
    store = ArtifactStore(StorageLayout(tmp_path / "store"), "geometry-v2")
    executor = ToolExecutor(registry, store)
    reconstruction = executor.execute(
        ToolCall(
            tool_name="reconstruct",
            arguments={
                "image_uris": [str(image)],
                "frame_indices": [17],
            },
        )
    )
    assert reconstruction.succeeded, reconstruction.error_message
    return image, mask, executor, store, reconstruction


def call(executor, name, **arguments):
    result = executor.execute(ToolCall(tool_name=name, arguments=arguments))
    assert result.succeeded, result.error_message
    return result


def test_reconstruction_contract_is_nonmetric_and_self_contained(scene):
    image, _, _, store, result = scene
    bundle = read_bundle(str(store.resolve(result.artifacts[0])))
    assert str(bundle["length_unit"]) == "reconstruction_unit"
    assert str(bundle["scale_status"]) == "unverified"
    assert result.structured_output["mean_confidence"] == pytest.approx(1.7)
    assert "confidence" not in result.metadata  # raw confidence is not calibrated
    assert bundle["source_image_uris"].tolist() == [str(image)]
    assert bundle["points"].shape == (1, 4, 6, 3)
    np.testing.assert_allclose(bundle["points"][0, 0, 0], [-2.5, -1.5, 2])
    np.testing.assert_allclose(
        bundle["source_to_processed"][0], [[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]]
    )


def test_centroid_and_point_artifact_feed_geometry(scene):
    image, mask, executor, store, reconstruction = scene
    result = call(
        executor,
        "mask",
        operation="masked_points",
        mask_uris=[str(mask)],
        source_image_uri=str(image),
        reconstruction_uri=reconstruction.artifacts[0].uri,
        frame_index=17,
    )
    assert result.structured_output["point_count"] == 24
    np.testing.assert_allclose(result.structured_output["centroid"], [0, 0, 2])
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    moved = call(
        executor,
        "geometry",
        operation="transform_points",
        points_uri=result.artifacts[0].uri,
        matrix=transform.tolist(),
        target_frame_id="translated",
        max_output_points=2,
    )
    assert moved.structured_output["points_truncated"]
    original = read_bundle(str(store.resolve(result.artifacts[0])))["points"]
    transformed = read_bundle(str(store.resolve(moved.artifacts[0])))["points"]
    np.testing.assert_allclose(transformed, original + [1, 2, 3])


def test_mask_source_mismatch_and_empty_mask(scene, tmp_path):
    _, mask, executor, _, reconstruction = scene
    result = executor.execute(
        ToolCall(
            tool_name="mask",
            arguments={
                "operation": "centroid_3d",
                "mask_uris": [str(mask)],
                "source_image_uri": str(tmp_path / "wrong.png"),
                "reconstruction_uri": reconstruction.artifacts[0].uri,
                "frame_index": 17,
            },
        )
    )
    assert not result.succeeded and result.error_type == "argument_validation"
    Image.new("L", (12, 8), 0).save(mask)
    empty = call(
        executor,
        "mask",
        operation="centroid_3d",
        mask_uris=[str(mask)],
        reconstruction_uri=reconstruction.artifacts[0].uri,
        frame_index=17,
    )
    assert not empty.structured_output["available"]
    assert empty.structured_output["centroid"] is None
    assert not empty.artifacts


def test_frame_indices_are_absolute(scene):
    _, mask, executor, _, reconstruction = scene
    result = executor.execute(
        ToolCall(
            tool_name="mask",
            arguments={
                "operation": "centroid_3d",
                "mask_uris": [str(mask)],
                "reconstruction_uri": reconstruction.artifacts[0].uri,
                "frame_index": 0,
            },
        )
    )
    assert not result.succeeded and "frame_index" in result.error_message


def test_projection_and_backprojection_round_trip(scene):
    _, _, executor, _, _ = scene
    c2w = np.eye(4)
    c2w[:3, 3] = [1, 2, 3]
    k = [[100, 0, 50], [0, 100, 40], [0, 0, 1]]
    projected = call(
        executor,
        "geometry",
        operation="project_points",
        points=[[2, 4, 7], [1, 2, 2]],
        camera_to_world=c2w.tolist(),
        intrinsics=k,
        length_unit="meter",
    )
    assert projected.structured_output["points"] == [[75, 90], None]
    assert projected.structured_output["valid"] == [True, False]
    recovered = call(
        executor,
        "geometry",
        operation="backproject_points",
        points=[[75, 90]],
        depth_values=[4],
        camera_to_world=c2w.tolist(),
        intrinsics=k,
        length_unit="meter",
    )
    np.testing.assert_allclose(recovered.structured_output["points"], [[2, 4, 7]])


def test_antiparallel_rotation_and_zero_vector_rejection(scene):
    _, _, executor, _, _ = scene
    result = call(
        executor,
        "geometry",
        operation="rotation_matrix_from_vectors",
        first=[1, 0, 0],
        second=[-1, 0, 0],
    )
    rotation = np.array(result.structured_output["rotation"])
    np.testing.assert_allclose(rotation @ [1, 0, 0], [-1, 0, 0])
    assert np.linalg.det(rotation) == pytest.approx(1)
    result = executor.execute(
        ToolCall(
            tool_name="geometry",
            arguments={
                "operation": "angle_between_vectors",
                "first": [0, 0, 0],
                "second": [1, 0, 0],
            },
        )
    )
    assert not result.succeeded and result.error_type == "argument_validation"


def test_cross_frame_distance_rejected(scene):
    _, _, executor, _, _ = scene
    result = executor.execute(
        ToolCall(
            tool_name="geometry",
            arguments={
                "operation": "point_distance",
                "first": [0, 0, 0],
                "second": [1, 1, 1],
                "first_frame_id": "one",
                "second_frame_id": "two",
            },
        )
    )
    assert not result.succeeded and "coordinate frame" in result.error_message


def test_scale_alignment_creates_new_artifact_without_mutation(scene):
    image, mask, executor, store, reconstruction = scene
    source_bytes = store.resolve(reconstruction.artifacts[0]).read_bytes()
    scaled = call(
        executor,
        "scale",
        image_uri=str(image),
        reconstruction_uri=reconstruction.artifacts[0].uri,
        frame_index=17,
        min_valid_pixels=8,
    )
    assert scaled.structured_output["alignment"]["correction_factor"] == pytest.approx(
        3
    )
    assert scaled.structured_output["intrinsics"][0][0] == pytest.approx(12)
    assert scaled.structured_output["intrinsics"][0][2] == pytest.approx(5.5)
    assert source_bytes == store.resolve(reconstruction.artifacts[0]).read_bytes()
    calibrated = next(
        a
        for a in scaled.artifacts
        if a.metadata.get("role") == "calibrated_reconstruction"
    )
    centroid = call(
        executor,
        "mask",
        operation="centroid_3d",
        mask_uris=[str(mask)],
        reconstruction_uri=calibrated.uri,
        frame_index=17,
    )
    assert centroid.structured_output["length_unit"] == "meter"
    assert centroid.structured_output["scale_status"] == "estimated_metric"
    np.testing.assert_allclose(centroid.structured_output["centroid"], [0, 0, 6])


def test_scale_disagreement_does_not_force_correction(scene):
    image, _, _, store, reconstruction = scene

    def predict(image_uri, resolution):
        depth = np.ones((8, 12), dtype=float)
        depth[:, 6:] = 100
        return {
            "depth": depth,
            "points": np.repeat(depth[..., None], 3, axis=-1),
            "intrinsics": np.eye(3),
        }

    tool = ScaleTool(MoGe2Adapter(predictor=predict))
    result = tool.execute(
        {
            "image_uri": str(image),
            "reconstruction_uri": str(store.resolve(reconstruction.artifacts[0])),
            "frame_index": 17,
            "min_valid_pixels": 8,
        },
        ToolContext(run_id="test"),
    )
    alignment = result.structured_output["alignment"]
    assert (
        alignment["reason"] == "high_disagreement"
        and alignment["correction_factor"] is None
    )
    assert not any(
        a.metadata.get("role") == "calibrated_reconstruction" for a in result.artifacts
    )


def test_object_frame_has_proper_axes_visible_obb_and_overlay(scene):
    image, mask, executor, _, reconstruction = scene
    pose = call(
        executor,
        "pose",
        operation="object_frame",
        image_uri=str(image),
        mask_uri=str(mask),
        reconstruction_uri=reconstruction.artifacts[0].uri,
        frame_index=17,
    )
    output = pose.structured_output
    assert output["available"] and output["orientation_status"] == "model_estimated"
    np.testing.assert_allclose(output["front"], [0, 0, 1])
    rotation = np.array(output["rotation"])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert len(output["obb_corners"]) == 8
    assert any(a.artifact_type is ArtifactType.IMAGE for a in pose.artifacts)


@pytest.mark.parametrize(
    "angles", [(0, 0, 0), (90, 0, 0), (30, 25, 40), (180, -89, -80)]
)
def test_orientation_projection_matches_upstream_plot_convention(angles):
    azimuth, polar, rotation = angles
    front, up = PoseTool._camera_axes(
        {"azimuth_deg": azimuth, "polar_deg": polar, "rotation_deg": rotation}
    )
    phi, theta, gamma = np.radians([azimuth, polar, -rotation])
    # Explicit formulas from Orient-Anything/utils.py:get_proj2D_XYZ (before drawing scale).
    front_plot = [
        -np.sin(phi) * np.cos(gamma) - np.cos(phi) * np.sin(theta) * np.sin(gamma),
        np.sin(phi) * np.sin(gamma) - np.cos(phi) * np.sin(theta) * np.cos(gamma),
    ]
    up_plot = [np.cos(theta) * np.sin(gamma), np.cos(theta) * np.cos(gamma)]
    np.testing.assert_allclose(front[:2] * [1, -1], front_plot, atol=1e-9)
    np.testing.assert_allclose(up[:2] * [1, -1], up_plot, atol=1e-9)
    assert np.dot(front, up) == pytest.approx(0, abs=1e-8)
    assert np.linalg.norm(front) == pytest.approx(1)


def test_degenerate_object_frame_is_unavailable(scene):
    image, mask, executor, _, reconstruction = scene
    result = call(
        executor,
        "pose",
        operation="object_frame",
        image_uri=str(image),
        mask_uri=str(mask),
        reconstruction_uri=reconstruction.artifacts[0].uri,
        frame_index=17,
        front_world=[0, 0, 1],
        up_world=[0, 0, 1],
    )
    assert not result.structured_output["available"]
    assert result.structured_output["reason"] == "degenerate_orientation_axes"


def test_graph_relations_are_visible_and_artifact_is_queryable(scene):
    _, _, executor, _, _ = scene
    result = call(
        executor,
        "graph",
        entities=[
            {"id": "a", "label": "first", "bbox": [0, 0, 1, 1], "depth_m": 1},
            {"id": "b", "label": "second", "bbox": [3, 0, 4, 1], "depth_m": 2},
        ],
        max_output_edges=1,
    )
    assert result.structured_output["edges"] == [
        {"source": "a", "relation": "left", "target": "b"}
    ]
    assert result.structured_output["edges_truncated"]
    queried = call(
        executor,
        "graph",
        graph_uri=result.artifacts[0].uri,
        relation="front",
        entity_id="a",
    )
    assert queried.structured_output["edges"] == [
        {"source": "a", "relation": "front", "target": "b"}
    ]


def test_graph_plot_produces_visible_image_and_ignores_invalid_values(scene):
    _, _, executor, store, _ = scene
    result = call(
        executor,
        "graph",
        operation="plot",
        values=[1, 100, None, 4],
        validity=[True, False, True, True],
        title="Valid depth",
        y_label="Depth (m)",
    )
    assert result.structured_output["valid_count"] == 2
    assert result.structured_output["mean"] == pytest.approx(2.5)
    assert result.structured_output["trend_slope_per_index"] == pytest.approx(1)
    assert result.artifacts[0].artifact_type is ArtifactType.IMAGE
    with Image.open(store.resolve(result.artifacts[0])) as chart:
        assert chart.size == (840, 480)


def test_moge_pointmap_can_directly_supply_masked_centroid(scene):
    image, mask, executor, _, _ = scene
    result = call(executor, "scale", image_uri=str(image))
    pointmap = next(
        a
        for a in result.artifacts
        if a.metadata.get("role") == "metric_camera_reconstruction"
    )
    centroid = call(
        executor,
        "mask",
        operation="centroid_3d",
        mask_uris=[str(mask)],
        reconstruction_uri=pointmap.uri,
        frame_index=0,
    )
    assert centroid.structured_output["length_unit"] == "meter"
    np.testing.assert_allclose(centroid.structured_output["centroid"], [6, 6, 6])


def test_mask_batch_medians_and_iou(scene, tmp_path):
    _, _, executor, _, _ = scene
    mask_a, mask_b = tmp_path / "a.png", tmp_path / "b.png"
    first = np.zeros((3, 5), dtype=np.uint8)
    first[0, [0, 1, 4]] = 255
    second = np.zeros_like(first)
    second[0, :2] = 255
    Image.fromarray(first).save(mask_a)
    Image.fromarray(second).save(mask_b)
    result = call(
        executor, "mask", operation="statistics", mask_uris=[str(mask_a), str(mask_b)]
    )
    assert result.structured_output["centroid_xy"] == [1, 0]
    assert len(result.structured_output["mask_statistics"]) == 2
    result = call(
        executor, "mask", operation="iou", mask_uris=[str(mask_a), str(mask_b)]
    )
    assert result.structured_output["iou"] == pytest.approx(2 / 3)


def test_multiview_points_agree_after_known_camera_translation(scene):
    image, _, _, _, _ = scene

    def predict(images):
        extrinsics = np.repeat(np.eye(4)[None], 2, axis=0)
        extrinsics[1, 0, 3] = -1
        return {
            "depth": np.full((2, 4, 6), 2),
            "conf": np.ones((2, 4, 6)),
            "intrinsics": np.repeat(np.diag([2, 2, 1])[None], 2, axis=0),
            "extrinsics": extrinsics,
        }

    result = ReconstructionTool(DepthAnything3Adapter(predictor=predict)).execute(
        {"image_uris": [str(image), str(image)]}, ToolContext(run_id="analytic")
    )
    import io

    with np.load(io.BytesIO(result.artifacts[0].data)) as bundle:
        np.testing.assert_allclose(
            bundle["points"][0, :, 1:], bundle["points"][1, :, :-1]
        )


def test_draw_lines_preserves_source_and_motion_empty_mask(scene, tmp_path):
    from spatialcraft.tools.builtin import DrawTool
    from spatialcraft.tools.real import FarnebackMotionTool

    image, _, _, _, _ = scene
    source_bytes = image.read_bytes()
    drawn = DrawTool().execute(
        {
            "image_uri": str(image),
            "lines": [{"line": [0, 0, 10, 7], "color": "red"}],
            "thickness": 2,
        },
        ToolContext(run_id="draw"),
    )
    assert drawn.structured_output["line_count"] == 1
    assert image.read_bytes() == source_bytes
    empty = tmp_path / "empty.png"
    Image.new("L", (12, 8), 0).save(empty)
    motion = FarnebackMotionTool().execute(
        {
            "first_image_uri": str(image),
            "second_image_uri": str(image),
            "mask_uri": str(empty),
        },
        ToolContext(run_id="flow"),
    )
    assert not motion.structured_output["available"]
    assert motion.structured_output["median_flow"] is None


def test_missing_reconstruction_confidence_is_not_fabricated(scene):
    import io

    image, _, _, _, _ = scene
    tool = ReconstructionTool(
        DepthAnything3Adapter(
            predictor=lambda images: {
                "depth": np.ones((1, 4, 6)),
                "intrinsics": np.eye(3)[None],
                "extrinsics": np.eye(4)[None],
            }
        )
    )
    result = tool.execute(
        {"image_uris": [str(image)]}, ToolContext(run_id="no-confidence")
    )
    assert result.structured_output["mean_confidence"] is None
    assert not result.structured_output["confidence_available"]
    with np.load(io.BytesIO(result.artifacts[0].data)) as bundle:
        assert "confidence" not in bundle.files
        assert bundle["valid"].all()
