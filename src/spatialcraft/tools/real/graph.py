"""Scene-graph construction from shared detection/depth coordinates."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..schema_builder import array_schema, object_schema

_ENTITY_SCHEMA = object_schema(
    {
        "id": {"type": "string", "minLength": 1},
        "label": {"type": "string", "minLength": 1},
        "bbox": array_schema({"type": "number"}, min_items=4, max_items=4),
        "depth_m": {"type": "number"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    required=("id", "label", "bbox"),
)


class SceneGraphTool(SpatialTool):
    spec = ToolSpec(
        name="graph",
        description="Build pairwise spatial-relation edges from detections and optional metric depth.",
        input_schema=object_schema(
            {
                "entities": array_schema(_ENTITY_SCHEMA, min_items=1),
                "frame_id": {"type": "string", "minLength": 1},
                "near_threshold": {"type": "number", "exclusiveMinimum": 0},
                "depth_threshold_m": {"type": "number", "minimum": 0},
            },
            required=("entities",),
        ),
    )

    @staticmethod
    def _center(entity: Mapping[str, Any]) -> tuple[float, float]:
        x1, y1, x2, y2 = (float(value) for value in entity["bbox"])
        return (x1 + x2) / 2, (y1 + y2) / 2

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        entities = [dict(item) for item in arguments["entities"]]
        near_threshold = float(arguments.get("near_threshold", 100.0))
        depth_threshold = float(arguments.get("depth_threshold_m", 0.1))
        edges: list[dict[str, Any]] = []
        for first, second in itertools.permutations(entities, 2):
            ax, ay = self._center(first)
            bx, by = self._center(second)
            relations = ["left" if ax < bx else "right" if ax > bx else "aligned-x"]
            relations.append(
                "above" if ay < by else "below" if ay > by else "aligned-y"
            )
            if math.dist((ax, ay), (bx, by)) <= near_threshold:
                relations.append("near")
            if "depth_m" in first and "depth_m" in second:
                delta = float(first["depth_m"]) - float(second["depth_m"])
                if abs(delta) > depth_threshold:
                    relations.append("behind" if delta > 0 else "front")
            for relation in relations:
                edges.append(
                    {
                        "source": first["id"],
                        "relation": relation,
                        "target": second["id"],
                    }
                )
        frame_id = str(arguments.get("frame_id", "scene:image"))
        graph = {"entities": entities, "edges": edges, "frame_id": frame_id}
        frame = CoordinateFrame(
            frame_id=frame_id,
            unit="pixel",
            convention="origin=top-left,x=right,y=down; optional depth in meters",
        )
        return ToolExecution(
            text=f"Built scene graph with {len(entities)} entities and {len(edges)} edges.",
            structured_output={"entity_count": len(entities), "edge_count": len(edges)},
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.SCENE_GRAPH,
                    json_value=graph,
                    suffix=".json",
                    mime_type="application/json",
                    frame_id=frame_id,
                ),
            ),
            coordinate_frames=(frame,),
            confidence=min(
                (float(entity.get("confidence", 1.0)) for entity in entities),
                default=1.0,
            ),
            unit="pixel",
        )


__all__ = ["SceneGraphTool"]
