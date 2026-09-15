"""CPU checks for the preflight/execution boundary and resume identity."""

import pytest

from memp import run_memp
from spatialcraft.storage.atomic_io import read_json


@pytest.fixture
def prepared(monkeypatch):
    calls = []
    manifest = {"datasets": {"robospatial": {"environment_count": 2, "count": 1}}}
    monkeypatch.setattr(run_memp, "prepare_inputs", lambda *args: manifest)
    monkeypatch.setattr(
        run_memp, "source_binding", lambda project: {"code_sha256": "fixture"}
    )
    monkeypatch.setattr(
        run_memp,
        "tool_paths_preflight",
        lambda: {"paths": {}, "missing": [], "root": "/fixture"},
    )
    monkeypatch.setattr(
        run_memp,
        "resource_binding",
        lambda *args: pytest.fail("Preflight hashed tool models"),
    )
    monkeypatch.setattr(
        run_memp,
        "load_api_key_file",
        lambda *args: pytest.fail("Preflight accessed credentials"),
    )
    monkeypatch.setattr(
        run_memp,
        "OpenAIEmbeddingsProvider",
        lambda *args, **kw: pytest.fail("Preflight constructed embedding client"),
    )
    monkeypatch.setattr(
        run_memp,
        "ScriptBuilder",
        lambda *args, **kw: pytest.fail("Preflight constructed script provider"),
    )
    return calls


def test_preflight_requires_no_gpu_api_or_credentials_and_can_resume(
    tmp_path, prepared
):
    args = ["--output", str(tmp_path / "run"), "--datasets", "robospatial"]
    assert run_memp.main(args) == 0
    report = read_json(tmp_path / "run/preflight.json")
    assert report["executes_api"] is False
    assert report["settings"]["memory_format"] == "proceduralization"
    assert report["counts"] == {"robospatial": {"environment": 2, "deployment": 1}}
    assert run_memp.main([*args, "--stage", "deployment"]) == 0
    with pytest.raises(ValueError, match="Resume binding changed"):
        run_memp.main([*args, "--top-k", "10"])


@pytest.mark.parametrize(
    "args",
    [
        ["--top-k", "0"],
        ["--temperature", "nan"],
        ["--environment-rollouts", "-1"],
        ["--datasets", "robospatial", "robospatial"],
    ],
)
def test_invalid_parameters_fail_before_preparing_data(tmp_path, monkeypatch, args):
    monkeypatch.setattr(
        run_memp,
        "prepare_inputs",
        lambda *args: pytest.fail("Invalid config touched inputs"),
    )
    with pytest.raises(SystemExit) as error:
        run_memp.main(["--output", str(tmp_path), *args])
    assert error.value.code == 2


def test_source_binding_covers_memp_and_reused_rag_data(tmp_path):
    paths = [
        "memp/runner.py",
        "memp/run_memp.py",
        "RAG/__init__.py",
        "RAG/data.py",
        "src/example.py",
        "prompts/system.md",
        "configs/test.yaml",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    before = run_memp.source_binding(tmp_path)
    assert set(paths) == set(before["source_files"])
    (tmp_path / "memp/runner.py").write_text("changed")
    assert run_memp.source_binding(tmp_path)["code_sha256"] != before["code_sha256"]
