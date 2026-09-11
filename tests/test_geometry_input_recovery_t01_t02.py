"""Real geometry dispatch and scripted production Runtime input-error recovery."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_runtime_v2 import runtime, tasks

from spatialcraft.models import (
    ModelProvider,
    ModelResponse,
    ResponseToolCall,
    TokenUsage,
)
from spatialcraft.schemas import ToolCall, Trajectory, TrajectoryStatus
from spatialcraft.storage import StorageLayout
from spatialcraft.tools import ArtifactStore, ToolExecutor, ToolRegistry
from spatialcraft.tools.builtin import GeometryTool
from spatialcraft.tools.spatial_arrays import camera_intrinsics, pixel_mapping, se3

SINGULAR_K = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
VALID_K = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


@pytest.fixture
def executor(tmp_path):
    registry = ToolRegistry()
    registry.register(GeometryTool())
    return ToolExecutor(
        registry, ArtifactStore(StorageLayout(tmp_path / "store"), "geometry-input")
    )


def geometry(executor, **arguments):
    return executor.execute(ToolCall(tool_name="geometry", arguments=arguments))


@pytest.mark.parametrize("operation", ["project_points", "backproject_points"])
@pytest.mark.parametrize(
    "intrinsics", [SINGULAR_K, [[1e-320, 0, 0], [0, 1, 0], [0, 0, 1]]]
)
def test_invalid_intrinsics_are_parameter_errors_in_both_directions(
    executor, operation, intrinsics
):
    result = geometry(
        executor,
        operation=operation,
        intrinsics=intrinsics,
        camera_to_world=np.eye(4).tolist(),
        points=[[0, 0]] if operation == "backproject_points" else [[0, 0, 2]],
        depth_values=[2],
    )
    assert not result.succeeded
    assert result.error_type == "argument_validation"
    assert "intrinsics" in result.error_message
    assert not result.artifacts


def test_general_affine_pixel_basis_still_round_trips(executor):
    intrinsics = [[2, 1, 10], [1, 3, 20], [0, 0, 1]]
    projected = geometry(
        executor,
        operation="project_points",
        points=[[2, 4, 2]],
        intrinsics=intrinsics,
        camera_to_world=np.eye(4).tolist(),
    )
    assert projected.succeeded and projected.structured_output["points"] == [[14, 27]]
    recovered = geometry(
        executor,
        operation="backproject_points",
        points_uri=projected.artifacts[0].uri,
        depth_values=[2],
    )
    assert recovered.succeeded, recovered.error_message
    np.testing.assert_allclose(recovered.structured_output["points"], [[2, 4, 2]])


@pytest.mark.parametrize(
    "normalize,dimension", [(se3, 4), (camera_intrinsics, 3), (pixel_mapping, 3)]
)
def test_near_homogeneous_rows_are_canonicalized_before_inverse(normalize, dimension):
    value = np.eye(dimension)
    value[0, -1] = 1e8
    value[-1, 0] = 1e-8
    before = value.copy()
    assert np.linalg.det(value) == 0
    normalized = normalize(value)
    np.testing.assert_array_equal(value, before)  # Never mutate caller calibration.
    np.testing.assert_array_equal(normalized[-1], np.eye(dimension)[-1])
    assert np.linalg.det(normalized) == pytest.approx(1)
    assert np.isfinite(np.linalg.inv(normalized)).all()
    value[-1, 0] = 0.01
    from spatialcraft.tools.schema_builder import ToolSchemaError

    with pytest.raises(ToolSchemaError):
        normalize(value)


def test_near_homogeneous_pose_project_artifact_backproject_is_consistent(executor):
    pose = np.eye(4)
    pose[0, 3] = 1e8
    pose[3, 0] = 1e-8
    projected = geometry(
        executor,
        operation="project_points",
        points=[[1e8, 0, 2]],
        intrinsics=VALID_K,
        camera_to_world=pose.tolist(),
    )
    assert projected.succeeded, projected.error_message
    assert projected.structured_output["points"] == [[0, 0]]
    assert projected.structured_output["camera_to_world"][3] == [0, 0, 0, 1]
    recovered = geometry(
        executor,
        operation="backproject_points",
        points_uri=projected.artifacts[0].uri,
        depth_values=[2],
    )
    assert recovered.succeeded, recovered.error_message
    np.testing.assert_allclose(recovered.structured_output["points"], [[1e8, 0, 2]])


@pytest.mark.parametrize(
    "operation,first,second",
    [
        ("point_distance", [0, 0], [3, 4]),
        ("point_distance", [0, 0, 0], [3, 4, 0]),
        ("bbox_relation", [0, 0, 1, 1], [3, 4, 5, 6]),
    ],
)
def test_operands_must_share_declared_frame_in_2d_and_3d(
    executor, operation, first, second
):
    for declarations in (
        {"first_frame_id": "camera-A", "second_frame_id": "camera-B"},
        {
            "frame_id": "camera-A",
            "first_frame_id": "camera-B",
            "second_frame_id": "camera-B",
        },
        {"source_frame_id": "camera-A", "second_frame_id": "camera-B"},
    ):
        result = geometry(
            executor, operation=operation, first=first, second=second, **declarations
        )
        assert not result.succeeded and result.error_type == "argument_validation"
        assert "coordinate frame" in result.error_message
    result = geometry(
        executor,
        operation=operation,
        first=first,
        second=second,
        first_frame_id="camera-A",
        second_frame_id="camera-A",
    )
    assert result.succeeded, result.error_message
    assert result.structured_output["source_frame_id"] == "camera-A"
    if operation == "point_distance":
        assert result.structured_output["distance"] == 5
    else:
        assert result.structured_output["relations"] == ["left", "above", "disjoint"]


class CountedGeometry(GeometryTool):
    def __init__(self):
        self.calls = []

    def execute(self, arguments, context):
        self.calls.append(dict(arguments))
        return super().execute(arguments, context)


class CorrectingModel(ModelProvider):
    def __init__(self, *, interrupt=False, valid_first=False):
        self.requests = []
        self.interrupt = interrupt
        self.valid_first = valid_first

    def generate(self, request):
        self.requests.append(request)
        assert request.metadata["operation"] == "execution.fallback"
        observations = [
            json.loads(message.text_content)
            for message in request.messages
            if message.role.value == "tool"
        ]
        if observations and observations[-1]["status"] == "succeeded":
            assert observations[-1]["structured_output"]["points"] == [[0, 0, 2]]
            return ModelResponse(
                provider="script",
                model=request.model_alias,
                text="Final Answer: yes",
                finish_reason="stop",
                usage=TokenUsage(input_tokens=2, output_tokens=2),
            )
        if observations:
            assert observations[-1]["status"] == "failed"
            assert "intrinsics must be invertible" in observations[-1]["error"]
            if self.interrupt:
                self.interrupt = False
                raise RuntimeError(
                    "fixture interruption after committed parameter error"
                )
        intrinsics = VALID_K if observations or self.valid_first else SINGULAR_K
        return ModelResponse(
            provider="script",
            model=request.model_alias,
            finish_reason="tool_calls",
            usage=TokenUsage(input_tokens=2, output_tokens=2),
            tool_calls=(
                ResponseToolCall(
                    name="geometry",
                    arguments={
                        "operation": "backproject_points",
                        "points": [[0, 0]],
                        "depth_values": [2],
                        "intrinsics": intrinsics,
                        "camera_to_world": np.eye(4).tolist(),
                    },
                ),
            ),
        )


def corrective_runtime(tmp_path, model):
    actual = runtime(tmp_path, model)
    # Keep the production pipeline/rollout path and isolate geometry correction.
    actual.settings = replace(
        actual.settings,
        experiment_name="geometry_input_recovery",
        ablations={"no_experience": True, "no_skill": True},
    )
    tool = CountedGeometry()
    actual.tools = ToolRegistry()
    actual.tools.register(tool)
    return actual, tool, (tasks(tmp_path)[0],)


@pytest.mark.parametrize("interrupt", [False, True])
def test_formal_runtime_corrects_invalid_intrinsics_and_resumes_without_reexecution(
    tmp_path, interrupt
):
    model = CorrectingModel(interrupt=interrupt)
    actual, tool, rows = corrective_runtime(tmp_path, model)
    pipeline = actual.dataset("fixture")
    first_result = None
    if interrupt:
        with pytest.raises(RuntimeError, match="fixture interruption"):
            pipeline.accumulate(rows)
        assert len(tool.calls) == 1
        first_result = pipeline.journal.read_committed(
            "tasks/00000/rollouts/00/execution/steps/0000/tools/000"
        )
        assert first_result["error_type"] == "argument_validation"
        pipeline = actual.dataset("fixture")
    frozen = pipeline.accumulate(rows)
    assert (
        len(tool.calls) == 8
    )  # One rejected and one corrected call in each of four rollouts.
    assert len(model.requests) == 12 + int(interrupt)
    for index in range(4):
        trajectory = Trajectory.from_dict(
            pipeline.journal.read_committed(
                f"tasks/00000/rollouts/{index:02d}/complete"
            )
        )
        assert (
            trajectory.status is TrajectoryStatus.COMPLETED and trajectory.reward == 1
        )
        assert len(trajectory.transitions) == 3
        assert (
            trajectory.transitions[0].tool_results[0].error_type
            == "argument_validation"
        )
        assert trajectory.transitions[1].tool_results[0].succeeded
        if index == 0 and first_result is not None:
            assert (
                trajectory.transitions[0].tool_results[0].result_id
                == first_result["result_id"]
            )
    assert actual.dataset("fixture").accumulate(rows).snapshot_id == frozen.snapshot_id
    assert len(tool.calls) == 8 and len(model.requests) == 12 + int(interrupt)


def test_formal_runtime_does_not_reclassify_numerical_backend_memory_failure(
    tmp_path, monkeypatch
):
    actual, tool, rows = corrective_runtime(tmp_path, CorrectingModel(valid_first=True))

    def allocation_failure(value):
        raise MemoryError("fixture numerical backend allocation failure")

    monkeypatch.setattr(np.linalg, "inv", allocation_failure)
    with pytest.raises(RuntimeError, match="Tool geometry failed: MemoryError"):
        actual.dataset("fixture").accumulate(rows)
    assert len(tool.calls) == 1
    assert len(actual.local.requests) == 1
