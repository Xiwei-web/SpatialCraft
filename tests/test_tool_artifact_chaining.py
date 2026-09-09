import io

from PIL import Image

from spatialcraft.schemas import ArtifactType, ToolCall, ToolStatus
from spatialcraft.storage import StorageLayout
from spatialcraft.tools import ArtifactStore, ToolExecutor
from spatialcraft.tools.base import ArtifactPayload
from spatialcraft.tools.real import create_real_tool_registry


def test_stored_mask_can_be_used_as_next_tool_input(tmp_path):
    store = ArtifactStore(StorageLayout(tmp_path / "store"), "chain")
    data = io.BytesIO()
    Image.new("L", (4, 4), 255).save(data, format="PNG")
    artifact = store.put(
        ArtifactPayload(
            artifact_type=ArtifactType.MASK,
            data=data.getvalue(),
            suffix=".png",
            mime_type="image/png",
        )
    )
    call = ToolCall(
        tool_name="mask", arguments={"mask_uris": [artifact.uri], "operation": "invert"}
    )
    result = ToolExecutor(create_real_tool_registry(), store).execute(call)
    assert result.status is ToolStatus.SUCCEEDED
    assert result.structured_output["foreground_pixels"] == 0
    assert call.arguments["mask_uris"] == [artifact.uri]
    assert result.metadata["resolved_artifact_uris"][artifact.uri] == str(
        store.layout.resolve_uri(artifact.uri)
    )


def test_invalid_geometry_is_recoverable_model_argument_error(tmp_path):
    store = ArtifactStore(StorageLayout(tmp_path / "store"), "invalid_box")
    call = ToolCall(
        tool_name="geometry",
        arguments={
            "operation": "bbox_relation",
            "first": [0, 0, 1, 1],
            "second": [8, 2, 5, 9],
        },
    )
    result = ToolExecutor(create_real_tool_registry(), store).execute(call)
    assert result.status is ToolStatus.FAILED
    assert result.error_type == "argument_validation"
    assert "maxima" in result.error_message
