"""Read-only committed-progress inspector; never loads credentials or labels."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--dataset", default="robospatial", choices=("robospatial", "erqa", "omni3d")
    )
    args = parser.parse_args()
    root = args.run / args.dataset
    if not (root / "journal.json").is_file():
        print(json.dumps({"status": "no_experiment_journal", "path": str(root)}))
        return
    journal = json.loads((root / "journal.json").read_text())
    report = {
        "run": str(args.run),
        "binding_sha256": journal["binding_sha256"],
        "training_rollouts_committed": len(
            list((root / "stages/tasks").glob("*/rollouts/*/complete/result.json"))
        ),
        "experience_updates_committed": len(
            list((root / "stages/tasks").glob("*/experience_update/result.json"))
        ),
        "evolution_rounds_committed": len(
            list((root / "stages/evolution").glob("*/skill_evolution/result.json"))
        ),
        "deployment_rollouts_committed": len(
            list((root / "stages/deployment").glob("*/complete/result.json"))
        ),
        "frozen_training_snapshot": (
            root / "checkpoints/frozen_deployment.json"
        ).is_file(),
        "final_result_exists": (root / "results/deployment.json").is_file(),
    }
    patch_file = root / "code_patch.json"
    for label, pattern in (
        ("experience_updates_skipped_output_limit", "tasks/*/experience_update/result.json"),
        ("evolution_rounds_skipped_output_limit", "evolution/*/skill_evolution/result.json"),
    ):
        report[label] = sum(
            json.loads(path.read_text())["result"].get("output_budget_recovery", {}).get("status")
            == "skipped_output_limit"
            for path in (root / "stages").glob(pattern)
        )
    if patch_file.is_file():
        patch = json.loads(patch_file.read_text())
        report["code_patch"] = {
            "reason": patch["reason"],
            "patched_snapshot": patch["patched_snapshot"],
            "active_code_sha256": patch["new_code_sha256"],
            "original_binding_sha256": patch["old_binding_sha256"],
        }
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    for path in (root / "stages/model_calls").glob("*/result.json"):
        value = json.loads(path.read_text())["result"].get("usage") or {}
        usage["calls"] += 1
        for key in ("input_tokens", "output_tokens"):
            usage[key] += value.get(key) or 0
    report["committed_local_generation_usage"] = usage
    logs = sorted(args.run.glob("launcher_logs/*.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        report["latest_launcher_log"] = str(logs[-1])
    if report["final_result_exists"]:
        report["result"] = json.loads((root / "results/deployment.json").read_text())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
