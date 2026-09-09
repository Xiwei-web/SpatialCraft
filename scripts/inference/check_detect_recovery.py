"""Replay the failed training detection with real weights, without any LLM/API."""

import argparse
from pathlib import Path

from spatialcraft.schemas import ToolCall
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import atomic_write_json, read_json
from spatialcraft.tools import ArtifactStore, ToolExecutor, ToolRegistry
from spatialcraft.tools.builtin.draw import render_annotations
from spatialcraft.tools.real.detect import GroundingDINOAdapter, GroundingDINOTool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    call = ToolCall.from_dict(read_json(args.inputs)["inputs"]["call"])
    values = call.arguments
    adapter = GroundingDINOAdapter()
    raw = adapter.predict(
        values["image_uri"],
        values["queries"],
        values.get("threshold", 0.35),
        values.get("text_threshold", 0.25),
    )
    report = {"source_inputs": str(args.inputs), "raw_detections": raw}
    try:
        render_annotations(values["image_uri"], boxes=raw)
        report["old_renderer_error"] = None
    except ValueError as exc:
        report["old_renderer_error"] = str(exc)
    registry = ToolRegistry()
    registry.register(GroundingDINOTool(GroundingDINOAdapter(predictor=lambda *a: raw)))
    result = ToolExecutor(
        registry,
        ArtifactStore(
            StorageLayout(args.output.parent / "artifacts"), "detect_recovery"
        ),
    ).execute(call)
    report["result"] = result.to_dict()
    report["status"] = "passed" if result.succeeded else "failed"
    atomic_write_json(args.output, report)
    print(report["status"], report["old_renderer_error"], flush=True)
    if not result.succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
