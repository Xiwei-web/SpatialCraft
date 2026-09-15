"""Import a verified environment prefix into a new, explicitly changed run."""

from pathlib import Path

from spatialcraft.experiments.journal import digest
from spatialcraft.models.serialization import request_to_dict
from spatialcraft.storage.atomic_io import file_lock, read_json

from .core import make_request


def model_key(task_index, rollout_index):
    return f"environment/tasks/{task_index:05d}/rollouts/{rollout_index:02d}/model"


def continuation_spec(source, datasets, completed_calls):
    if len(datasets) != 1 or completed_calls < 1:
        raise ValueError(
            "Environment continuation requires one dataset and a positive prefix"
        )
    source = Path(source).resolve()
    manifest = read_json(source / datasets[0] / "journal.json")
    binding = manifest["binding"]
    if manifest["binding_sha256"] != digest(binding):
        raise ValueError("Source journal binding checksum mismatch")
    if binding["datasets"] != list(datasets):
        raise ValueError("Source dataset selection differs")
    return {
        "policy": "retain_committed_prefix_then_remaining_task_budget_v1",
        "source_run": str(source),
        "source_journal_sha256": digest(manifest),
        "completed_calls": completed_calls,
        "source_environment_rollouts": binding["options"]["environment_rollouts"],
    }


def rollout_counts(task_count, options):
    budget = options["environment_rollouts"]
    spec = options.get("environment_resume")
    if not spec:
        return [budget] * task_count
    old_budget = spec["source_environment_rollouts"]
    completed = spec["completed_calls"]
    if not 1 <= budget < old_budget or not 0 < completed < task_count * old_budget:
        raise ValueError(
            "Continuation must reduce the remaining budget of a partial run"
        )
    # Started tasks already meeting the new budget need no additional generation.
    return [
        max(budget, min(old_budget, max(0, completed - i * old_budget)))
        for i in range(task_count)
    ]


def _comparable(binding):
    value = {k: v for k, v in binding.items() if k != "code_sha256"}
    value["options"] = {
        k: v
        for k, v in value["options"].items()
        if k not in {"environment_rollouts", "environment_resume"}
    }
    return value


def import_prefix(journal, tasks, identity, config, options):
    """Validate source commits and exact requests; never request imported responses."""
    spec = options.get("environment_resume")
    if not spec:
        return
    rollout_counts(len(tasks), options)
    source_run = Path(spec["source_run"])
    source = source_run / tasks[0].dataset
    if source.resolve() == journal.root.resolve():
        raise ValueError("Changed sampling requires a separate output directory")

    def perform_import():
        with (
            file_lock(source_run / ".rag_run.lock", timeout=0),
            file_lock(source / ".rag.lock", timeout=0),
        ):
            manifest = read_json(source / "journal.json")
            if digest(manifest) != spec["source_journal_sha256"] or manifest[
                "binding_sha256"
            ] != digest(manifest["binding"]):
                raise ValueError(
                    "Source journal changed after continuation was configured"
                )
            if _comparable(manifest["binding"]) != _comparable(journal.binding):
                raise ValueError(
                    "Continuation changed model, data or another non-budget setting"
                )
            if (
                manifest["binding"]["options"]["environment_rollouts"]
                != spec["source_environment_rollouts"]
            ):
                raise ValueError("Source rollout budget mismatch")
            keys = [
                model_key(*divmod(i, spec["source_environment_rollouts"]))
                for i in range(spec["completed_calls"])
            ]
            actual = {
                str(p.parent.relative_to(source / "stages"))
                for p in (source / "stages/environment/tasks").glob(
                    "*/rollouts/*/model/result.json"
                )
            }
            if actual != set(keys):
                raise ValueError(
                    "Source commits do not match the requested contiguous prefix"
                )
            checksums = {}
            for i, key in enumerate(keys):
                task_index, rollout_index = divmod(
                    i, spec["source_environment_rollouts"]
                )
                request = make_request(
                    tasks[task_index],
                    config,
                    stage="environment",
                    rollout_index=rollout_index,
                )
                expected_inputs = {
                    "request": request_to_dict(request),
                    "media_sha256": {
                        part.uri: identity["media"][part.uri]
                        for message in request.messages
                        for part in message.content
                        if part.uri
                    },
                }
                unit = source / "stages" / key
                envelope = read_json(unit / "inputs.json")
                commit = read_json(unit / "result.json")
                if (
                    envelope["binding_sha256"] != manifest["binding_sha256"]
                    or commit["binding_sha256"] != manifest["binding_sha256"]
                    or digest(envelope["inputs"]) != digest(expected_inputs)
                    or envelope["inputs_sha256"] != digest(expected_inputs)
                    or commit["inputs_sha256"] != digest(expected_inputs)
                    or commit["result_sha256"] != digest(commit["result"])
                ):
                    raise ValueError(
                        f"Source request or result checksum mismatch: {key}"
                    )
                journal.execute(
                    key, expected_inputs, lambda value=commit["result"]: value
                )
                checksums[key] = commit["result_sha256"]
            return {
                "source": spec,
                "imported_calls": len(keys),
                "result_sha256": checksums,
            }

    journal.execute("environment/import_prefix", {"source": spec}, perform_import)
