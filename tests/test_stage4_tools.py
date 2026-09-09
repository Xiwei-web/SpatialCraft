from __future__ import annotations

from pathlib import Path

from PIL import Image

from spatialcraft.models import ModelResponse, ResponseToolCall, to_agent_action
from spatialcraft.schemas import ArtifactType, ToolCall, ToolStatus
from spatialcraft.storage import StorageLayout
from spatialcraft.tools import (
    ArtifactPayload,
    ArtifactStore,
    MockDetectionTool,
    RemoteSpatialTool,
    ToolContext,
    ToolExecution,
    ToolExecutor,
    create_mock_tool_registry,
)


def _image(path: Path) -> Path:
    Image.new("RGB", (64, 48), color=(80, 120, 160)).save(path)
    return path


def _executor(tmp_path: Path) -> tuple[ToolExecutor, ArtifactStore]:
    store = ArtifactStore(StorageLayout(tmp_path / "storage"), "test-run")
    return ToolExecutor(create_mock_tool_registry(), store), store


def test_model_tool_call_executes_and_persists_outputs(tmp_path) -> None:
    image = _image(tmp_path / "input.png")
    executor, store = _executor(tmp_path)
    response = ModelResponse(
        provider="mock",
        model="mock-model",
        tool_calls=(
            ResponseToolCall(
                name="detect",
                arguments={"image_uri": str(image), "queries": ["chair", "table"]},
            ),
        ),
    )
    action = to_agent_action(response)
    results = executor.execute_action(
        action, context=ToolContext(run_id="test-run", task_id="task-1")
    )
    assert len(results) == 1
    result = results[0]
    assert result.status is ToolStatus.SUCCEEDED
    assert result.text
    assert {item.artifact_type for item in result.artifacts} == {
        ArtifactType.BOUNDING_BOXES,
        ArtifactType.IMAGE,
    }
    assert result.coordinate_frames[0].unit == "pixel"
    assert 0 <= result.metadata["confidence"] <= 1
    assert result.metadata["unit"] == "pixel"
    assert all(store.resolve(item).is_file() for item in result.artifacts)
    assert store.read_result(result.result_id) == result


def test_invalid_json_arguments_become_audited_failure(tmp_path) -> None:
    executor, store = _executor(tmp_path)
    result = executor.execute(
        ToolCall(
            tool_name="detect",
            arguments={"image_uri": "missing.png", "queries": []},
        )
    )
    assert result.status is ToolStatus.FAILED
    assert result.error_type == "argument_validation"
    assert "queries" in (result.error_message or "")
    assert store.read_result(result.result_id) == result


def test_depth_geometry_and_draw_return_durable_artifacts(tmp_path) -> None:
    image = _image(tmp_path / "input.png")
    executor, store = _executor(tmp_path)
    calls = (
        ToolCall(tool_name="depth", arguments={"image_uri": str(image)}),
        ToolCall(
            tool_name="geometry",
            arguments={
                "operation": "bbox_relation",
                "first": [0, 0, 20, 20],
                "second": [30, 10, 50, 30],
            },
        ),
        ToolCall(
            tool_name="draw",
            arguments={
                "image_uri": str(image),
                "points": [{"point": [12, 10], "label": "target"}],
            },
        ),
    )
    results = tuple(executor.execute(call) for call in calls)
    assert all(result.succeeded for result in results)
    assert results[0].metadata["unit"] == "meter"
    assert results[1].structured_output["relations"] == [
        "left",
        "above",
        "disjoint",
    ]
    assert all(
        store.resolve(item).exists() for result in results for item in result.artifacts
    )
    assert len(store.iter_results()) == 3


def test_same_public_tool_contract_supports_remote_backend(tmp_path) -> None:
    image = _image(tmp_path / "input.png")
    registry = create_mock_tool_registry()
    spec = MockDetectionTool.spec

    def transport(request):
        assert request["tool"] == "detect"
        assert request["arguments"]["queries"] == ["lamp"]
        return ToolExecution(
            text="Remote detection complete.",
            structured_output={"detections": []},
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.JSON,
                    json_value={"detections": []},
                    suffix=".json",
                    mime_type="application/json",
                ),
            ),
            confidence=1.0,
            unit="pixel",
            metadata={"service": "fake"},
        ).to_wire()

    registry.register(
        RemoteSpatialTool(spec, "https://tool.invalid/detect", transport=transport),
        backend="remote",
    )
    store = ArtifactStore(StorageLayout(tmp_path / "storage"), "remote-run")
    result = ToolExecutor(registry, store).execute(
        ToolCall(
            tool_name="detect",
            arguments={"image_uri": str(image), "queries": ["lamp"]},
        ),
        backend="remote",
    )
    assert result.succeeded
    assert result.metadata["backend"] == "remote"
    assert result.metadata["service"] == "fake"
    assert store.resolve(result.artifacts[0]).is_file()
