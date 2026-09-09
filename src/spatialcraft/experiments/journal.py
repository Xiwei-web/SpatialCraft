"""Crash-safe, input-bound, idempotent stage commits for future experiment runners."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatialcraft.storage.atomic_io import (
    atomic_write_json,
    canonical_json_bytes,
    file_lock,
    read_json,
    sha256_bytes,
)
from spatialcraft.storage.layout import safe_relative_path


def digest(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _json_value(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy into its durable JSON representation before comparison or hashing."""

    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(
                    f"Duplicate journal JSON key after normalization: {key}"
                )
            result[key] = item
        return result

    # JSON object keys are strings. Normalize before canonical sorting: HF
    # max_memory legitimately mixes integer GPU indices with the "cpu" key.
    # Do not change the original dictionary passed to the model loader.
    return json.loads(
        json.dumps(dict(value), allow_nan=False, ensure_ascii=False),
        object_pairs_hook=unique_object,
    )


def _execution_revision(previous, overrides):
    """Only the explicitly requested BF16 two-GPU/FLA backend may change."""
    if not overrides:
        return previous
    if set(overrides) != {"model_local", "kernel_runtime"}:
        raise ValueError("Unsupported execution revision fields")
    old_local = previous["model_local"]
    expected_local = {
        **old_local,
        "device_map": "balanced",
        "extra_load_kwargs": {
            **old_local["extra_load_kwargs"],
            "max_memory": {"0": "28GiB", "1": "34GiB"},
        },
    }
    kernel = overrides["kernel_runtime"]
    if (
        overrides["model_local"] != expected_local
        or kernel.get("profile") != "qwen35_9b_fla_two_gpu_v1"
        or kernel.get("expected_gpus") != 2
        or not kernel.get("installation_sha256")
        or old_local.get("dtype") != "bfloat16"
        or old_local.get("load_in_4bit") is not False
    ):
        raise ValueError("Unsupported two-GPU kernel execution revision")
    return {**previous, **overrides}


class RunJournal:
    """Immutable stage result is the sole commit point; incomplete work is retried.

    Runtime callbacks should perform one reproducible stage, e.g. a model call,
    tool call, verification or batch knowledge snapshot. They must NOT mutate a
    shared knowledge bank in place: commit the complete next snapshot as result.
    A remote call can repeat if interrupted before its durable result is written;
    this is not an exactly-once guarantee for external side effects.
    """

    def __init__(self, root: str | Path, binding: Mapping[str, Any]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = _json_value(binding)
        self.binding_digest = digest(self.binding)
        self.accepted_binding_digests = {self.binding_digest}
        self.binding_ranks = {self.binding_digest: 0}
        with file_lock(self.root / ".manifest.lock"):
            manifest = self.root / "journal.json"
            expected = {
                "schema_version": 1,
                "binding": self.binding,
                "binding_sha256": self.binding_digest,
            }
            if manifest.exists():
                original = read_json(manifest)
                patch_path = self.root / "code_patch.json"
                if patch_path.exists():
                    # Explicit chained bugfix lineage, not a generic bypass.
                    # Data, weights, sampling and all other bindings MUST match.
                    patch = read_json(patch_path)
                    old = original["binding"]
                    old_digest = digest(old)
                    chain = [*patch.get("prior_patches", []), patch]
                    previous = old
                    historical_digests = {old_digest}
                    chain_valid = True
                    historical_ranks = {old_digest: 0}
                    for index, revision in enumerate(chain, start=1):
                        next_binding = {
                            **_execution_revision(
                                previous, revision.get("execution_overrides", {})
                            ),
                            "code_sha256": revision["new_code_sha256"],
                        }
                        if (
                            revision.get("schema_version") != 1
                            or revision.get("old_binding_sha256") != old_digest
                            or not revision.get("reason")
                            or not revision.get("source_changes")
                            or next_binding == previous
                            or (
                                revision.get(
                                    "parent_code_sha256",
                                    previous["code_sha256"] if index == 1 else None,
                                )
                                != previous["code_sha256"]
                            )
                            or digest(next_binding) in historical_digests
                        ):
                            chain_valid = False
                        historical_digests.add(digest(next_binding))
                        historical_ranks[digest(next_binding)] = index
                        previous = next_binding
                    patched = previous
                    if (
                        original
                        != {
                            "schema_version": 1,
                            "binding": old,
                            "binding_sha256": old_digest,
                        }
                        or patch.get("schema_version") != 1
                        or patch.get("old_binding_sha256") != old_digest
                        or not patch.get("reason")
                        or not patch.get("source_changes")
                        or patched == old
                        or self.binding != patched
                        or not chain_valid
                    ):
                        raise ValueError("Invalid code-patch resume binding")
                    self.accepted_binding_digests.update(historical_digests)
                    self.binding_ranks = historical_ranks
                elif original != expected:
                    raise ValueError("Resume binding changed; use a new run directory")
            else:
                atomic_write_json(manifest, expected, overwrite=False)

    def _unit(self, key: str) -> Path:
        relative = safe_relative_path(key, "stage key")
        path = self.root / "stages" / relative
        if not path.resolve().is_relative_to(self.root / "stages"):
            raise ValueError("Stage path escapes journal")
        return path

    def read_committed(self, key: str) -> dict[str, Any]:
        """Read a required prior commit without creating or executing a stage."""
        unit = self._unit(key)
        envelope = read_json(unit / "inputs.json")
        commit = read_json(unit / "result.json")
        if (
            envelope["binding_sha256"] not in self.accepted_binding_digests
            or digest(envelope["inputs"]) != envelope["inputs_sha256"]
            or commit["inputs_sha256"] != envelope["inputs_sha256"]
            or commit["binding_sha256"] not in self.accepted_binding_digests
            or (
                self.binding_ranks[commit["binding_sha256"]]
                < self.binding_ranks[envelope["binding_sha256"]]
            )
            or digest(commit["result"]) != commit["result_sha256"]
        ):
            raise ValueError("Prior stage checksum/binding mismatch")
        return commit["result"]

    def execute(
        self,
        key: str,
        inputs: Mapping[str, Any],
        operation: Callable[[], Mapping[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        """Return (committed result, reused); preserve failed-attempt diagnostics."""
        unit = self._unit(key)
        unit.mkdir(parents=True, exist_ok=True)
        input_value = _json_value(inputs)
        input_digest = digest(input_value)
        with file_lock(unit / ".lock", timeout=0):
            input_path = unit / "inputs.json"
            envelope = {
                "binding_sha256": self.binding_digest,
                "inputs_sha256": input_digest,
                "inputs": input_value,
            }
            if input_path.exists():
                previous = read_json(input_path)
                if (
                    previous.get("binding_sha256") not in self.accepted_binding_digests
                    or {**previous, "binding_sha256": self.binding_digest} != envelope
                ):
                    raise ValueError(
                        "Stage inputs changed; refusing incompatible resume"
                    )
            else:
                atomic_write_json(input_path, envelope, overwrite=False)
            commit_path = unit / "result.json"
            if commit_path.exists():
                commit = read_json(commit_path)
                if (
                    commit["inputs_sha256"] != input_digest
                    or commit["binding_sha256"] not in self.accepted_binding_digests
                    or (
                        self.binding_ranks[commit["binding_sha256"]]
                        < self.binding_ranks[read_json(input_path)["binding_sha256"]]
                    )
                    or digest(commit["result"]) != commit["result_sha256"]
                ):
                    raise ValueError("Stage result checksum/binding mismatch")
                return commit["result"], True
            attempts = unit / "attempts"
            attempts.mkdir(exist_ok=True)
            number = max((int(p.stem) for p in attempts.glob("*.json")), default=0) + 1
            attempt_path = attempts / f"{number:06d}.json"
            attempt = {
                "attempt": number,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(attempt_path, attempt, overwrite=False)
            print(f"[{attempt['started_at']}] START {key}", flush=True)
            try:
                result = _json_value(operation())
                commit = {
                    "binding_sha256": self.binding_digest,
                    "inputs_sha256": input_digest,
                    "result_sha256": digest(result),
                    "result": result,
                }
                # This atomic create, not a progress counter, marks completion.
                atomic_write_json(commit_path, commit, overwrite=False)
            except BaseException as exc:
                attempt.update(
                    status="interrupted"
                    if isinstance(exc, (KeyboardInterrupt, SystemExit))
                    else "failed",
                    error_type=type(exc).__name__,
                )
                atomic_write_json(attempt_path, attempt)
                print(f"FAILED {key}: {type(exc).__name__}", flush=True)
                raise
            attempt.update(
                status="completed", completed_at=datetime.now(timezone.utc).isoformat()
            )
            atomic_write_json(attempt_path, attempt)
            print(f"[{attempt['completed_at']}] COMMIT {key}", flush=True)
            return result, False
