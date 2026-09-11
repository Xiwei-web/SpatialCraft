"""Scene-graph construction from shared detection/depth coordinates."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

from spatialcraft.schemas import ArtifactType, CoordinateFrame

from ..base import ArtifactPayload, SpatialTool, ToolContext, ToolExecution, ToolSpec
from ..schema_builder import ToolSchemaError, array_schema, enum_schema, object_schema

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
        version="2.0.0",
        description="operation=plot plots numerical values (optional validity) as an image with statistics. Otherwise build image-plane bbox relations and optional camera-depth ordering, or query an existing graph_uri. Output edges are bounded; graph_uri plus entity_id/relation filters lets later calls consume the full graph artifact.",
        input_schema=object_schema(
            {
                "operation": enum_schema("scene_graph", "query", "plot"),
                "values": array_schema(
                    {"type": ["number", "null"]}, min_items=1, max_items=10000
                ),
                "validity": array_schema(
                    {"type": "boolean"}, min_items=1, max_items=10000
                ),
                "x_label": {"type": "string"},
                "y_label": {"type": "string"},
                "title": {"type": "string"},
                "entities": array_schema(_ENTITY_SCHEMA, min_items=1, max_items=64),
                "graph_uri": {"type": "string", "minLength": 1},
                "entity_id": {"type": "string", "minLength": 1},
                "relation": {"type": "string", "minLength": 1},
                "max_output_edges": {"type": "integer", "minimum": 1, "maximum": 512},
                "offset": {"type": "integer", "minimum": 0},
                "frame_id": {"type": "string", "minLength": 1},
                "near_threshold": {"type": "number", "exclusiveMinimum": 0},
                "depth_threshold_m": {"type": "number", "minimum": 0},
            },
            required=(),
        ),
    )

    @staticmethod
    def _center(entity: Mapping[str, Any]) -> tuple[float, float]:
        x1, y1, x2, y2 = (float(value) for value in entity["bbox"])
        return (x1 + x2) / 2, (y1 + y2) / 2

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        if arguments.get("operation") == "plot":
            return self._plot(arguments, context)
        if "graph_uri" in arguments:
            import json
            from pathlib import Path

            graph = json.loads(Path(arguments["graph_uri"]).read_text())
            edges = graph["edges"]
            if "entity_id" in arguments:
                edges = [
                    e
                    for e in edges
                    if arguments["entity_id"] in (e["source"], e["target"])
                ]
            if "relation" in arguments:
                edges = [e for e in edges if e["relation"] == arguments["relation"]]
            offset, limit = (
                int(arguments.get("offset", 0)),
                int(arguments.get("max_output_edges", 128)),
            )
            return ToolExecution(
                text=f"Graph query: {len(edges)} matching edges.",
                structured_output={
                    "edges": edges[offset : offset + limit],
                    "matching_edge_count": len(edges),
                    "offset": offset,
                    "edges_truncated": offset + limit < len(edges),
                    "frame_id": graph["frame_id"],
                },
            )
        if "entities" not in arguments:
            raise ToolSchemaError("graph requires entities or graph_uri")
        entities = [dict(item) for item in arguments["entities"]]
        if len({entity["id"] for entity in entities}) != len(entities):
            raise ToolSchemaError("graph entity IDs must be unique")
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
        edge_limit = int(arguments.get("max_output_edges", 128))
        frame = CoordinateFrame(
            frame_id=frame_id,
            unit="pixel",
            convention="origin=top-left,x=right,y=down; optional depth in meters",
        )
        return ToolExecution(
            text=f"Built scene graph with {len(entities)} entities and {len(edges)} edges.",
            structured_output={
                "entity_count": len(entities),
                "edge_count": len(edges),
                "entities": entities,
                "edges": edges[:edge_limit],
                "edges_truncated": len(edges) > edge_limit,
                "frame_id": frame_id,
                "relation_semantics": {
                    "left_right_above_below": "image-plane bbox centers",
                    "near": "image-plane pixel distance",
                    "front_behind": "camera depth ordering when both entities have depth_m",
                },
                "near_threshold_pixels": near_threshold,
                "depth_threshold_m": depth_threshold,
            },
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

    @staticmethod
    def _plot(arguments: Mapping[str, Any], context: ToolContext) -> ToolExecution:
        import io

        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        if "values" not in arguments:
            raise ToolSchemaError("graph plot requires values")
        values = np.asarray(
            [np.nan if value is None else value for value in arguments["values"]],
            dtype=float,
        )
        validity = np.asarray(
            arguments.get("validity", np.ones(len(values), dtype=bool)), dtype=bool
        )
        if validity.shape != values.shape:
            raise ToolSchemaError("plot validity must match values")
        validity &= np.isfinite(values)
        samples = np.flatnonzero(validity)
        finite = values[validity]
        slope = float(np.polyfit(samples, finite, 1)[0]) if len(samples) > 1 else None
        output = {
            "operation": "plot",
            "sample_count": len(values),
            "valid_count": len(finite),
            "minimum": float(finite.min()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
            "mean": float(finite.mean()) if len(finite) else None,
            "trend_slope_per_index": slope,
            "trend": "unavailable"
            if slope is None
            else "flat"
            if abs(slope) < 1e-12
            else "increasing"
            if slope > 0
            else "decreasing",
        }
        figure = Figure(figsize=(7, 4), dpi=120)
        canvas = FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(
            np.arange(len(values)),
            np.where(validity, values, np.nan),
            color="#3067b2",
            marker="." if len(values) <= 100 else None,
        )
        axis.set_xlabel(str(arguments.get("x_label", "Frame")))
        axis.set_ylabel(str(arguments.get("y_label", "Value")))
        axis.set_title(str(arguments.get("title", "Numerical sequence")))
        axis.grid(True, alpha=0.25)
        if not len(finite):
            axis.text(
                0.5, 0.5, "No valid samples", transform=axis.transAxes, ha="center"
            )
        figure.tight_layout()
        data = io.BytesIO()
        canvas.print_png(data)
        frame = CoordinateFrame(
            frame_id=f"plot:{context.invocation_id}",
            unit="pixel",
            convention="rendered plot image; axis values labeled separately",
        )
        return ToolExecution(
            text=f"Sequence plot: {output}",
            structured_output=output,
            artifacts=(
                ArtifactPayload(
                    artifact_type=ArtifactType.IMAGE,
                    data=data.getvalue(),
                    suffix=".png",
                    mime_type="image/png",
                    shape=(480, 840, 4),
                    dtype="uint8",
                    frame_id=frame.frame_id,
                    metadata={
                        "role": "numerical_sequence_plot",
                        "x_label": arguments.get("x_label", "Frame"),
                        "y_label": arguments.get("y_label", "Value"),
                    },
                ),
            ),
            coordinate_frames=(frame,),
        )


__all__ = ["SceneGraphTool"]
