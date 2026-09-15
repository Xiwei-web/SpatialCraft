"""Offline checks of MemRL's preflight boundary and variant isolation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from MemRL import run_memrl
from spatialcraft.storage.atomic_io import read_json


def forbidden(message):
    def fail(*args, **kwargs):
        pytest.fail(message)

    return fail


@pytest.fixture
def prepared(monkeypatch):
    manifest = {"datasets": {"robospatial": {"environment_count": 2, "count": 1}}}
    monkeypatch.setattr(run_memrl, "prepare_inputs", lambda *args: manifest)
    monkeypatch.setattr(
        run_memrl, "source_binding", lambda _: {"code_sha256": "fixture"}
    )
    monkeypatch.setattr(
        run_memrl,
        "tool_paths_preflight",
        lambda: {
            "paths": {},
            "missing": [],
            "root": "/fixture",
        },
    )
    monkeypatch.setattr(
        run_memrl,
        "create_real_tool_registry",
        lambda: SimpleNamespace(
            definitions=list,
        ),
    )
    for name, message in (
        ("resource_binding", "Preflight read GPU model checkpoint bytes"),
        ("load_api_key_file", "Preflight accessed credentials"),
        ("OpenAIEmbeddingsProvider", "Preflight constructed an embedding client"),
        ("ReflectionBuilder", "Preflight constructed a reflection client"),
        ("DatasetRunner", "Preflight constructed an executing runner"),
    ):
        monkeypatch.setattr(run_memrl, name, forbidden(message))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=forbidden("Preflight accessed CUDA"),
            )
        ),
    )
    return manifest


def cli_args(tmp_path, variant="both"):
    return [
        "--variant",
        variant,
        "--output",
        str(tmp_path / "run"),
        "--datasets",
        "robospatial",
    ]


def test_default_preflight_has_no_external_calls_and_allows_stage_resume(
    tmp_path, prepared
):
    args = cli_args(tmp_path)
    assert run_memrl.main(args) == 0
    report = read_json(tmp_path / "run/preflight.json")
    assert report["executes_api"] is False
    assert report["variants"] == ["R", "GT"]
    assert report["counts"] == {"robospatial": {"environment": 2, "deployment": 1}}
    assert report["settings"]["model"] == "gpt-5.4-mini"
    assert report["generation"]["reasoning_effort"] == "medium"
    assert report["generation"]["max_output_tokens"] == 16384
    assert report["settings"]["similarity_threshold"] is None
    assert run_memrl.main([*args, "--stage", "environment"]) == 0
    assert run_memrl.main([*args, "--stage", "deployment"]) == 0


@pytest.mark.parametrize(
    "changed",
    [
        ["--variant", "R"],
        ["--model", "gpt-5.4"],
        ["--reasoning-effort", "none"],
        ["--max-output-tokens", "8192"],
        ["--candidate-k", "20"],
        ["--utility-weight", "0.75"],
        ["--learning-rate", "0.6"],
        ["--q-init", "0.5"],
        ["--similarity-threshold", "0.7"],
        ["--threshold-quantile", "0.9"],
        ["--environment-passes", "2"],
        ["--environment-rollouts", "2"],
    ],
)
def test_resume_rejects_variant_or_method_changes(tmp_path, prepared, changed):
    args = cli_args(tmp_path)
    assert run_memrl.main(args) == 0
    with pytest.raises(ValueError, match="Resume binding changed"):
        run_memrl.main([*args, *changed])


@pytest.mark.parametrize(
    "extra",
    [
        ["--top-k", "0"],
        ["--candidate-k", "2"],
        ["--temperature", "nan"],
        ["--similarity-threshold", "inf"],
        ["--utility-weight", "1.1"],
        ["--environment-passes", "-1"],
        ["--datasets", "robospatial", "robospatial"],
        ["--stage", "build"],
    ],
)
def test_invalid_settings_fail_before_preparation(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(
        run_memrl, "prepare_inputs", forbidden("Invalid settings touched data")
    )
    with pytest.raises(SystemExit) as exc:
        run_memrl.main([*cli_args(tmp_path), *extra])
    assert exc.value.code == 2


def test_variant_is_required_and_source_output_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_memrl, "prepare_inputs", forbidden("Invalid arguments touched data")
    )
    with pytest.raises(SystemExit) as exc:
        run_memrl.main(["--output", str(tmp_path)])
    assert exc.value.code == 2
    source_alias = tmp_path / "source_alias"
    source_alias.symlink_to(run_memrl.PROJECT, target_is_directory=True)
    with pytest.raises(SystemExit) as exc:
        run_memrl.main(
            ["--variant", "R", "--output", str(source_alias / "unsafe_output")]
        )
    assert exc.value.code == 2


def test_both_execution_uses_isolated_variant_builders_and_output_roots(
    tmp_path,
    monkeypatch,
    prepared,
):
    executions, builders, caches, closed, credential_calls = [], [], [], [], []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
            )
        ),
    )
    monkeypatch.setattr(
        run_memrl, "resource_binding", lambda _: {"tool_code": "fixture"}
    )
    monkeypatch.setattr(
        run_memrl, "load_api_key_file", lambda path: credential_calls.append(path)
    )
    splits = {
        "environment": (["public-env"], ["private-env"]),
        "deployment": (["public-test"], ["private-test"]),
    }
    monkeypatch.setattr(
        run_memrl, "load_split", lambda output, name, identity, phase: splits[phase]
    )

    class Embedding:
        def __init__(self, config, cache_dir):
            self.cache_dir = Path(cache_dir)
            caches.append(self.cache_dir)
            self._client = SimpleNamespace(close=lambda: closed.append(self.cache_dir))

    class Builder:
        def __init__(self, config, variant):
            self.config, self.variant = config, variant
            builders.append(self)

        def close(self):
            closed.append(self.variant)

    class Runner:
        def __init__(
            self,
            output,
            name,
            identity,
            settings,
            config,
            resources,
            *,
            variant,
            embedder,
            builder,
            registry,
        ):
            assert builder.variant == variant
            assert builder.config is config
            assert embedder.cache_dir == Path(output) / name / "embedding_cache"
            assert resources["code_sha256"] == "fixture"
            self.variant, self.output = variant, output
            executions.append(self)

        def run(self, stage, *, environment, deployment):
            assert stage == "all"
            assert environment is splits["environment"]
            assert deployment is splits["deployment"]
            return {"status": "completed", "variant": self.variant}

    monkeypatch.setattr(run_memrl, "OpenAIEmbeddingsProvider", Embedding)
    monkeypatch.setattr(run_memrl, "ReflectionBuilder", Builder)
    monkeypatch.setattr(run_memrl, "DatasetRunner", Runner)
    assert run_memrl.main([*cli_args(tmp_path), "--execute"]) == 0
    assert [item.variant for item in executions] == ["R", "GT"]
    assert [item.output for item in executions] == [
        tmp_path / "run/R",
        tmp_path / "run/GT",
    ]
    assert builders[0] is not builders[1]
    assert len(set(caches)) == 2
    assert len(credential_calls) == 1
    assert len(closed) == 4
    report = read_json(tmp_path / "run/results/all_summary.json")
    assert report["status"] == "completed"
    assert set(report["variants"]) == {"R", "GT"}
    assert all(set(value) == {"robospatial"} for value in report["variants"].values())


def test_source_binding_covers_all_reused_code_and_ignores_run_records(tmp_path):
    paths = [
        "MemRL/run_memrl.py",
        "MemRL/runner.py",
        "MemRL/memory.py",
        "MemRL/reflection.py",
        "MemRL/run_memrl.sh",
        "MemRL/memrl.sbatch",
        "memp/runner.py",
        "memp/data.py",
        "RAG/__init__.py",
        "RAG/data.py",
        "src/example.py",
        "prompts/system.md",
        "configs/test.yaml",
    ]
    for name in [*paths, "MemRL/runs/report.md", "MemRL/tests/test_fixture.py"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    before = run_memrl.source_binding(tmp_path)
    assert set(before["source_files"]) == set(paths)
    (tmp_path / "MemRL/runs/report.md").write_text("new report")
    assert run_memrl.source_binding(tmp_path) == before
    (tmp_path / "memp/data.py").write_text("changed shared input validation")
    assert run_memrl.source_binding(tmp_path)["code_sha256"] != before["code_sha256"]
