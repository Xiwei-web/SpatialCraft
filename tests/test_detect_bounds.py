import copy
import json

from PIL import Image

from spatialcraft.tools.base import ToolContext
from spatialcraft.tools.coordinate_frames import validate_bbox
from spatialcraft.tools.real.detect import (
    GroundingDINOAdapter,
    GroundingDINOTool,
    sanitize_detections,
)


def detection(box):
    return {"bbox": box, "label": "cabinet", "confidence": 0.9}


def test_detector_clipping_and_invalid_filter():
    boxes = [
        detection(b)
        for b in (
            [1, 2, 30, 40],
            [-5, -4, 105, 85],
            [1, 2, 100, 80],
            [101, 2, 130, 40],
            [30, 2, 10, 40],
            [10, 2, 10, 40],
            [0, 0, float("nan"), 40],
            [-20, 1, 0, 20],
        )
    ]
    original = copy.deepcopy(boxes[:6])
    valid, audit = sanitize_detections(boxes, 100, 80)
    assert len(valid) == 3
    assert valid[0] == boxes[0]
    assert valid[1]["bbox"] == [0, 0, 99, 79]
    assert valid[2]["bbox"] == [1, 2, 99, 79]
    assert len(audit) == 7
    assert boxes[:6] == original
    json.dumps(audit, allow_nan=False)
    for item in valid:
        validate_bbox(item["bbox"], width=100, height=80)


def test_detector_renders_clipped_boxes_and_records_original(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (100, 80)).save(path)
    adapter = GroundingDINOAdapter(
        predictor=lambda *args: [
            detection([200, 2, 230, 40]),
            detection([-2, 3, 100, 80]),
        ]
    )
    result = GroundingDINOTool(adapter).execute(
        {"image_uri": str(path), "queries": ["cabinet"], "max_detections": 1},
        ToolContext(run_id="test"),
    )
    assert result.structured_output["detections"][0]["bbox"] == [0, 3, 99, 79]
    assert result.structured_output["bbox_postprocessing"]["changes"][1][
        "raw_bbox"
    ] == [-2, 3, 100, 80]
    assert len(result.artifacts) == 2


def test_empty_detector_output_is_not_infrastructure_failure(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (100, 80)).save(path)
    for predictions in ([], [detection([200, 2, 230, 40])]):
        adapter = GroundingDINOAdapter(
            predictor=lambda *args, predictions=predictions: predictions
        )
        result = GroundingDINOTool(adapter).execute(
            {"image_uri": str(path), "queries": ["cabinet"]},
            ToolContext(run_id="test"),
        )
        assert result.structured_output["detections"] == []
        assert result.confidence == 0
