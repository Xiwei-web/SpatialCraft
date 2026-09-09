from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from spatialcraft.schemas import ToolCall
from spatialcraft.storage import StorageLayout
from spatialcraft.tools import ArtifactStore, ToolExecutor, ToolRegistry
from spatialcraft.tools.builtin import DrawTool, GeometryTool
from spatialcraft.tools.real import (
    DepthAnything3Adapter,
    EasyOCRAdapter,
    FarnebackMotionTool,
    GroundingDINOAdapter,
    GroundingDINOTool,
    MaskTool,
    MoGe2Adapter,
    OCRTool,
    OrientAnythingAdapter,
    PoseTool,
    ReconstructionTool,
    SAM3Adapter,
    SAM3Tool,
    ScaleTool,
    SceneGraphTool,
    SpatialToolPaths,
    create_real_tool_registry,
)


def _images(tmp_path: Path) -> tuple[Path, Path]:
    first = np.zeros((48, 64, 3), dtype=np.uint8)
    second = first.copy()
    cv2.rectangle(first, (8, 12), (24, 28), (255, 255, 255), -1)
    cv2.rectangle(second, (12, 12), (28, 28), (255, 255, 255), -1)
    first_path, second_path = tmp_path / "first.png", tmp_path / "second.png"
    Image.fromarray(first).save(first_path)
    Image.fromarray(second).save(second_path)
    return first_path, second_path


def test_real_registry_is_complete_and_lazy() -> None:
    paths = SpatialToolPaths()
    required = (
        paths.groundingdino_checkpoint,
        paths.sam3_checkpoint,
        paths.moge_checkpoint,
        paths.da3_checkpoint / "model.safetensors",
        paths.orient_checkpoint,
        paths.easyocr_models / "craft_mlt_25k.pth",
    )
    assert all(path.is_file() for path in required)
    registry = create_real_tool_registry()
    assert registry.names() == (
        "detect",
        "draw",
        "geometry",
        "graph",
        "mask",
        "motion",
        "ocr",
        "pose",
        "reconstruct",
        "scale",
        "segment",
    )
    assert all(
        not registry.get(name).adapter.loaded
        for name in ("detect", "segment", "scale", "reconstruct", "pose", "ocr")
    )


def _injected_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        GroundingDINOTool(
            GroundingDINOAdapter(
                predictor=lambda image, queries, box, text: [
                    {
                        "label": queries[0],
                        "bbox": [8, 12, 24, 28],
                        "confidence": 0.92,
                    }
                ]
            )
        )
    )

    def segment(image, prompt, boxes, labels, threshold):
        mask = np.zeros((48, 64), dtype=bool)
        mask[12:29, 8:25] = True
        return {"masks": mask[None], "boxes": boxes, "scores": np.array([0.94])}

    registry.register(SAM3Tool(SAM3Adapter(predictor=segment)))
    registry.register(MaskTool())
    registry.register(GeometryTool())

    def scale(image, resolution):
        depth = np.full((48, 64), 2.0, dtype=np.float32)
        yy, xx = np.mgrid[:48, :64]
        points = np.stack((xx / 32, yy / 32, depth), axis=-1).astype(np.float32)
        return {"points": points, "depth": depth, "intrinsics": np.eye(3)}

    registry.register(ScaleTool(MoGe2Adapter(predictor=scale)))

    def reconstruct(images):
        count = len(images)
        return {
            "depth": np.full((count, 48, 64), 2.0, dtype=np.float32),
            "conf": np.full((count, 48, 64), 0.8, dtype=np.float32),
            "intrinsics": np.repeat(np.eye(3)[None], count, axis=0),
            "extrinsics": np.repeat(np.eye(4)[None], count, axis=0),
        }

    registry.register(ReconstructionTool(DepthAnything3Adapter(predictor=reconstruct)))
    registry.register(
        PoseTool(
            OrientAnythingAdapter(
                predictor=lambda image, bbox: {
                    "azimuth_deg": 90.0,
                    "polar_deg": 5.0,
                    "rotation_deg": -10.0,
                    "confidence": 0.88,
                }
            )
        )
    )
    registry.register(SceneGraphTool())
    registry.register(FarnebackMotionTool())
    registry.register(
        OCRTool(
            EasyOCRAdapter(
                predictor=lambda image, languages: [
                    {
                        "polygon": [[5, 5], [30, 5], [30, 15], [5, 15]],
                        "text": "LEFT",
                        "confidence": 0.97,
                    }
                ]
            )
        )
    )
    registry.register(DrawTool())
    return registry


def test_all_real_tool_contracts_compose_and_persist(tmp_path) -> None:
    first, second = _images(tmp_path)
    registry = _injected_registry()
    store = ArtifactStore(StorageLayout(tmp_path / "storage"), "real-tools")
    executor = ToolExecutor(registry, store)
    detect = executor.execute(
        ToolCall(
            tool_name="detect",
            arguments={"image_uri": str(first), "queries": ["square"]},
        )
    )
    boxes = [item["bbox"] for item in detect.structured_output["detections"]]
    segment = executor.execute(
        ToolCall(
            tool_name="segment",
            arguments={"image_uri": str(first), "boxes": boxes},
        )
    )
    mask_uri = str(store.resolve(segment.artifacts[0]))
    calls = (
        ToolCall(
            tool_name="mask",
            arguments={"mask_uris": [mask_uri], "operation": "dilate"},
        ),
        ToolCall(
            tool_name="geometry",
            arguments={
                "operation": "bbox_relation",
                "first": boxes[0],
                "second": [30, 10, 50, 30],
            },
        ),
        ToolCall(tool_name="scale", arguments={"image_uri": str(first)}),
        ToolCall(
            tool_name="reconstruct",
            arguments={"image_uris": [str(first), str(second)]},
        ),
        ToolCall(
            tool_name="pose", arguments={"image_uri": str(first), "bbox": boxes[0]}
        ),
        ToolCall(
            tool_name="graph",
            arguments={
                "entities": [
                    {"id": "a", "label": "square", "bbox": boxes[0], "depth_m": 2.0},
                    {
                        "id": "b",
                        "label": "other",
                        "bbox": [30, 10, 50, 30],
                        "depth_m": 3.0,
                    },
                ]
            },
        ),
        ToolCall(
            tool_name="motion",
            arguments={
                "first_image_uri": str(first),
                "second_image_uri": str(second),
            },
        ),
        ToolCall(tool_name="ocr", arguments={"image_uri": str(first)}),
        ToolCall(
            tool_name="draw",
            arguments={"image_uri": str(first), "boxes": [{"bbox": boxes[0]}]},
        ),
    )
    results = (detect, segment, *(executor.execute(call) for call in calls))
    assert len(results) == 11
    assert all(result.succeeded for result in results), [
        (result.tool_name, result.error_message)
        for result in results
        if not result.succeeded
    ]
    assert all(result.text for result in results)
    assert all(result.coordinate_frames for result in results)
    assert all(
        store.resolve(artifact).is_file()
        for result in results
        for artifact in result.artifacts
    )
    assert len(store.iter_results()) == 11
    assert results[4].metadata["unit"] == "meter"
    assert results[5].structured_output["view_count"] == 2
    assert results[6].metadata["unit"] == "degree"
    assert results[9].structured_output["regions"][0]["text"] == "LEFT"
