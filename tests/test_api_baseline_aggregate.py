"""Three-run statistics must reject incomplete, incompatible or copied results."""

import importlib.util
import json
from pathlib import Path

import pytest

from spatialcraft.storage.atomic_io import atomic_write_json, sha256_file

_spec = importlib.util.spec_from_file_location(
    "api_aggregate",
    Path(__file__).resolve().parents[1]
    / "scripts/inference/aggregate_api_baselines.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.fixture(params=["gpt-5.4-mini", "gpt-5.4"])
def runs(tmp_path, request):
    roots = []
    for index, rewards in enumerate(((0, 0), (1, 0), (1, 1))):
        root = tmp_path / f"run{index}"
        roots.append(root)
        atomic_write_json(
            root / "results/summary.json",
            {"status": "completed", "failed_datasets": []},
        )
        atomic_write_json(
            root / "sat/journal.json",
            {
                "binding": {
                    "model_id": request.param,
                    "generation": {"temperature": 0, "seed": None},
                    "data_sha256": "fixed",
                }
            },
        )
        public = root / "data/sat/public.jsonl"
        public.parent.mkdir(parents=True)
        public.write_text(
            "".join(json.dumps({"task_id": f"t{i}"}) + "\n" for i in range(2))
        )
        atomic_write_json(
            root / "data/manifest.json",
            {
                "datasets": {
                    "sat": {
                        "task_ids": ["t0", "t1"],
                        "public_sha256": sha256_file(public),
                    }
                }
            },
        )
        rows = [
            {
                "task_id": f"t{i}",
                "response_id": f"resp_{index}_{i}",
                "reward": r,
                "model": request.param + "-2026-03-05",
                "question_type": "direction",
                "answer_type": "multiple_choice",
                "source_answer_type": "unknown",
            }
            for i, r in enumerate(rewards)
        ]
        atomic_write_json(
            root / "sat/results/deployment.json",
            {
                "status": "completed",
                "evaluated": 2,
                "total": 2,
                "correct": sum(rewards),
                "accuracy": sum(rewards) / 2,
                "model_ids": [request.param + "-2026-03-05"],
            },
        )
        (root / "sat/results/predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    return roots


def test_sample_std_and_variance(runs, tmp_path):
    result = _module.aggregate(runs, counts={"sat": 2})
    row = result["datasets"]["sat"]
    assert row["accuracy_runs"] == [0, 0.5, 1]
    assert row["mean"] == 0.5 and row["std"] == 0.5 and row["variance"] == 0.25
    assert row["std_population"] == pytest.approx((1 / 6) ** 0.5)
    assert row["categories"]["question_type"]["direction"]["std"] == 0.5
    assert result["api_seed"] is None
    assert result["model"] in {"gpt-5.4", "gpt-5.4-mini"}
    _module.write_results(tmp_path / "aggregate", result)
    assert (tmp_path / "aggregate/accuracy_mean_std.json").is_file()
    rendered = (tmp_path / "aggregate/accuracy_mean_std.md").read_text()
    assert "50.00 ± 50.00%" in rendered
    assert rendered.startswith(f"# {result['model']}: three repeated evaluations")


def test_requires_three_distinct_directories(runs):
    with pytest.raises(ValueError, match="three distinct"):
        _module.aggregate([runs[0]] * 3, counts={"sat": 2})


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "binding",
        "duplicate_response",
        "wrong_accuracy",
        "task_ids",
        "model_snapshot",
    ],
)
def test_reject_invalid_comparisons(runs, damage):
    root = runs[1]
    if damage == "incomplete":
        atomic_write_json(
            root / "results/summary.json",
            {"status": "failed", "failed_datasets": ["sat"]},
        )
    elif damage == "binding":
        atomic_write_json(
            root / "sat/journal.json", {"binding": {"generation": {"temperature": 0.7}}}
        )
    elif damage == "wrong_accuracy":
        path = root / "sat/results/deployment.json"
        value = json.loads(path.read_text())
        value["accuracy"] = 0.1
        atomic_write_json(path, value)
    else:
        path = root / "sat/results/predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "duplicate_response":
            rows[0]["response_id"] = "resp_0_0"
        elif damage == "task_ids":
            rows[0]["task_id"] = "changed"
        elif damage == "model_snapshot":
            rows[0]["model"] = "gpt-5.4-mini-changed"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        _module.aggregate(runs, counts={"sat": 2})
