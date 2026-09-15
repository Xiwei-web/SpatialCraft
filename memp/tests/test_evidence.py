"""Historical observations retain geometry while excluding run-specific audit."""

import json
from types import SimpleNamespace

from memp.runner import semantic_trajectory
from spatialcraft.schemas import (
    AgentAction,
    ArtifactRef,
    ArtifactType,
    ToolCall,
    ToolResult,
    ToolStatus,
)
from spatialcraft.storage.atomic_io import sha256_file


def test_semantic_trace_keeps_full_geometry_json_and_rewrites_historical_uris(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "edges": [{"relation": "left_of", "distance": 1.23}],
                "invocation_id": "private-run-id",
                "resolved_artifact_uris": {"runs/old/graph.json": str(path)},
            }
        )
    )
    image = str(tmp_path / "old.png")
    uri = "runs/old/graph.json"
    artifact = ArtifactRef(
        artifact_type=ArtifactType.JSON,
        uri=uri,
        mime_type="application/json",
        sha256=sha256_file(path),
        metadata={"invocation_id": "old-call", "length_unit": "meter"},
    )
    action = AgentAction.tool(
        ToolCall(tool_name="Geometry", arguments={"image_uri": image}),
        reasoning_summary=f"Measure objects in {image}",
    )
    result = ToolResult(
        tool_call_id=action.tool_calls[0].call_id,
        tool_name="Geometry",
        status=ToolStatus.SUCCEEDED,
        structured_output={"values": list(range(5000)), "invocation_id": "audit-only"},
        artifacts=(artifact,),
    )
    trajectory = SimpleNamespace(
        task=SimpleNamespace(images=[SimpleNamespace(uri=image)]),
        transitions=[
            SimpleNamespace(step_index=0, action=action, tool_results=(result,)),
            SimpleNamespace(
                step_index=1, action=AgentAction.final("A"), tool_results=()
            ),
        ],
    )
    trace = semantic_trajectory(
        trajectory, SimpleNamespace(resolve_uri=lambda unused: path)
    )
    serialized = json.dumps(trace)
    assert len(trace) == 2 and trace[-1]["final_answer"] == "A"
    assert trace[0]["observations"][0]["structured_output"]["values"] == list(
        range(5000)
    )
    historical = trace[0]["observations"][0]["artifacts"][0]
    assert historical["sha256"] == sha256_file(path)
    assert historical["json_observation"]["edges"][0]["distance"] == 1.23
    assert "MEMORY_IMAGE_1" in serialized and "MEMORY_ARTIFACT_" in serialized
    assert (
        image not in serialized
        and uri not in serialized
        and str(path) not in serialized
    )
    assert (
        "invocation_id" not in serialized and "resolved_artifact_uris" not in serialized
    )
