"""Replay committed knowledge responses offline; never regenerate model output."""

import argparse
import json
from pathlib import Path

from spatialcraft.experiments.learning import _json_object
from spatialcraft.storage.atomic_io import atomic_write_json, read_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calls = args.run / "robospatial/stages/model_calls"
    failed = (
        calls
        / "1436c2c8f88ea1c1b30f6ca54086201e597e83c8db0a45e958f316076f27c446/result.json"
    )
    before = sha256_file(failed)
    response = read_json(failed)["result"]
    assert response["finish_reason"] == "stop"
    parsed = _json_object(response["text"])
    assert set(parsed) == {"diagnosis", "initiation", "policy", "termination"}
    assert all(isinstance(v, str) and v for v in parsed.values())
    assert (
        parsed["termination"]
        == "Correct. The agent correctly terminated the action after the independent verification confirmed the conclusion."
    )
    matched = 0
    for path in calls.glob("*/result.json"):
        text = read_json(path)["result"].get("text") or ""
        try:
            valid = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(valid, dict):
            assert _json_object(text) == valid, path
            matched += 1
    assert sha256_file(failed) == before
    atomic_write_json(
        args.output,
        {
            "status": "passed",
            "kind": "offline_committed_response_replay",
            "failed_response_path": str(failed),
            "failed_response_sha256": before,
            "parsed_diagnosis": parsed,
            "unchanged_valid_json_objects": matched,
            "raw_response_modified": False,
            "model_or_paid_api_calls": 0,
        },
        overwrite=False,
    )
    print({"status": "passed", "unchanged_valid_json_objects": matched})


if __name__ == "__main__":
    main()
