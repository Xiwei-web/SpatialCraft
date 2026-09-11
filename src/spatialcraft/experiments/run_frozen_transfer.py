"""Deploy a checksum-validated frozen source bank with another local model.

Source journals are opened read-only. The target gets a separate transfer journal;
no source training stages are fabricated or rerun. Default CLI mode is offline
preflight; --execute is the existing explicit real-inference opt-in.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.storage.atomic_io import (
    atomic_write_json,
    file_lock,
    read_json,
    sha256_file,
)

from .accumulation import KnowledgeState
from .journal import _execution_revision, digest
from .operation_profiles import operation_profile, resolved_configuration
from .prepare_v2 import DATASETS
from .run import load_api_key_file, preflight
from .run_memory_baseline import bind_inference_resources


@dataclass(frozen=True)
class FrozenSource:
    root: Path
    knowledge: KnowledgeState
    dataset: str
    training_task_ids: tuple[str, ...]
    provenance: dict
    file_hashes: dict

    def verify_unchanged(self):
        for relative, checksum in self.file_hashes.items():
            if sha256_file(self.root / relative) != checksum:
                raise ValueError("Frozen source journal changed after validation")
        if digest(self.knowledge.to_dict()) != self.provenance["knowledge_sha256"]:
            raise ValueError("Frozen source knowledge object was modified")


def _source_binding_ranks(root, manifest):
    binding = manifest.get("binding")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(binding, dict)
        or digest(binding) != manifest.get("binding_sha256")
    ):
        raise ValueError("Source journal manifest checksum mismatch")
    original_digest, previous = digest(binding), binding
    ranks, files = {original_digest: 0}, ["journal.json"]
    patch_path = root / "code_patch.json"
    if patch_path.exists():
        patch = read_json(patch_path)
        files.append("code_patch.json")
        for index, revision in enumerate(
            [*patch.get("prior_patches", []), patch], start=1
        ):
            if (
                revision.get("schema_version") != 1
                or revision.get("old_binding_sha256") != original_digest
                or not revision.get("reason")
                or not revision.get("source_changes")
                or revision.get(
                    "parent_code_sha256",
                    previous.get("code_sha256") if index == 1 else None,
                )
                != previous.get("code_sha256")
            ):
                raise ValueError("Invalid source journal code-patch lineage")
            current = {
                **_execution_revision(
                    previous, revision.get("execution_overrides", {})
                ),
                "code_sha256": revision["new_code_sha256"],
            }
            identity = digest(current)
            if identity in ranks:
                raise ValueError("Repeated source journal revision")
            ranks[identity] = index
            previous = current
    return ranks, previous, files


def read_frozen_source(path, expected_snapshot_id=None):
    """Validate committed source data without constructing/writing a RunJournal."""
    root = Path(path).resolve()
    manifest = read_json(root / "journal.json")
    ranks, effective_binding, files = _source_binding_ranks(root, manifest)

    def committed(key):
        names = [f"stages/{key}/inputs.json", f"stages/{key}/result.json"]
        envelope, result = (read_json(root / name) for name in names)
        if (
            envelope.get("binding_sha256") not in ranks
            or result.get("binding_sha256") not in ranks
            or digest(envelope.get("inputs")) != envelope.get("inputs_sha256")
            or result.get("inputs_sha256") != envelope.get("inputs_sha256")
            or ranks[result["binding_sha256"]] < ranks[envelope["binding_sha256"]]
            or digest(result.get("result")) != result.get("result_sha256")
        ):
            raise ValueError(f"Source committed stage checksum/binding mismatch: {key}")
        files.extend(names)
        return envelope["inputs"], result["result"]

    initial_inputs, initial = committed("initial")
    final_inputs, final = committed("frozen_deployment_snapshot")
    knowledge = KnowledgeState.from_dict(final)
    if final_inputs.get("snapshot_id") != knowledge.snapshot_id:
        raise ValueError("Source final snapshot identity mismatch")
    if (
        expected_snapshot_id is not None
        and expected_snapshot_id != knowledge.snapshot_id
    ):
        raise ValueError("Source snapshot does not match the requested identity")
    dataset = initial.get("dataset")
    ids = initial.get("training_task_ids")
    if (
        dataset not in DATASETS
        or not isinstance(ids, list)
        or not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
        or initial_inputs.get("task_ids") != ids
        or effective_binding.get("dataset", dataset) != dataset
    ):
        raise ValueError("Source training dataset/task identity is inconsistent")
    settings = effective_binding.get("settings", {})
    roles = settings.get("roles", {})
    role_models = effective_binding.get("role_models", {})
    source_executor = roles.get("executor", settings.get("backbone"))
    source_kb = roles.get("knowledge_builder", settings.get("backbone"))
    if not source_executor or not source_kb:
        raise ValueError(
            "Source journal does not identify executor/knowledge-builder models"
        )
    hashes = {name: sha256_file(root / name) for name in files}
    provenance = {
        "source_journal": str(root),
        "source_snapshot_id": knowledge.snapshot_id,
        "source_binding_sha256": digest(effective_binding),
        "source_dataset": dataset,
        "source_training_task_ids": ids,
        "source_executor": source_executor,
        "source_knowledge_builder": source_kb,
        "source_role_models": role_models,
        "source_protocol_version": settings.get("protocol_version", "legacy_v1"),
        "source_method": settings.get("experiment_name", "full"),
        "knowledge_sha256": digest(final),
        "source_files_sha256": hashes,
    }
    return FrozenSource(root, knowledge, dataset, tuple(ids), provenance, hashes)


def validate_target_location(source, output):
    target = Path(output).resolve()
    if (
        target == source.root
        or target.is_relative_to(source.root)
        or source.root.is_relative_to(target)
    ):
        raise ValueError("Frozen transfer requires a separate target output directory")


def validate_target_tasks(source, tasks):
    from spatialcraft.schemas import TaskSplit

    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Transfer requires nonempty unique deployment tasks")
    if any(task.dataset != source.dataset for task in tasks):
        raise ValueError("Cross-model transfer must use the source benchmark")
    if any(task.split is not TaskSplit.TEST for task in tasks):
        raise ValueError("Transfer accepts deployment/test tasks only")
    if set(source.training_task_ids).intersection(task.task_id for task in tasks):
        raise ValueError("Transfer deployment overlaps source training task IDs")


def target_contract(project, settings, source, *, token_counter=None):
    if not settings.is_v2 or settings.backbone not in {
        "qwen3.5-9b",
        "qwen3.6-27b-local",
    }:
        raise ValueError(
            "Frozen transfer currently supports v2 local Qwen3.5-9B/Qwen3.6-27B executors only"
        )
    if settings.ablations:
        raise ValueError("Declare transfer settings without unrelated method ablations")
    config = ModelConfig.from_dict(
        load_yaml(Path(project) / f"configs/models/{settings.backbone}.yaml")
    )
    if config.local is None:
        raise ValueError("API-only frozen transfer is not implemented")
    tokenizer_path = config.local.tokenizer_path or config.local.path
    tokenizer_files = {
        str(path.relative_to(Path(tokenizer_path))): sha256_file(path)
        for path in Path(tokenizer_path).glob("*")
        if path.is_file() and path.suffix in {".json", ".jinja", ".txt", ".model"}
    }
    if token_counter is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
            revision=config.local.revision,
        )
        token_counter = lambda text: len(
            tokenizer.encode(text, add_special_tokens=False)
        )
    limit = int(settings.skill_options.get("stored_skill_max_tokens", 1024))
    lengths = {
        skill.reference: int(token_counter(skill.format_for_prompt()))
        for skill in source.knowledge.skills.active()
    }
    if any(length > limit for length in lengths.values()):
        raise ValueError(
            "Frozen source Skill exceeds target tokenizer limit; source will not be rewritten"
        )
    if (
        len(source.knowledge.experiences.active()) > settings.experience_capacity
        or len(source.knowledge.skills.active()) > settings.skill_capacity
    ):
        raise ValueError(
            "Frozen source knowledge exceeds the target's declared capacities"
        )
    resolved = resolved_configuration(settings)
    profile = operation_profile(settings, "execution.deployment")
    return {
        "target_executor": settings.backbone,
        "target_knowledge_builder": resolved["roles"]["knowledge_builder"],
        "target_tokenizer": {
            "path": str(tokenizer_path),
            "revision": config.local.revision,
            "files_sha256": tokenizer_files,
        },
        "target_reasoning_mode": profile.reasoning_mode,
        "target_generation_profile": resolved["operations"]["execution.deployment"],
        "target_skill_token_limit": limit,
        "target_skill_token_counts": lengths,
        "frozen_knowledge_rewritten": False,
        "target_settings": resolved,
    }


class FrozenTransferDeployment:
    """A separate transfer commit plus the Runtime's existing v2 deployment loop."""

    def __init__(self, source, target_pipeline, contract):
        self.source, self.pipeline, self.contract = source, target_pipeline, contract
        validate_target_location(source, target_pipeline.journal.root)
        if not target_pipeline.settings.is_v2:
            raise ValueError("Frozen transfer requires the v2 target deployment loop")
        if (target_pipeline.journal.root / "stages/initial/result.json").exists():
            raise ValueError(
                "Target journal already contains accumulation; use a new transfer run"
            )

    def deploy(self, tasks):
        tasks = tuple(tasks)
        validate_target_tasks(self.source, tasks)
        self.source.verify_unchanged()
        journal = self.pipeline.journal
        with file_lock(journal.root / ".frozen_transfer.lock", timeout=0):
            imported, _ = journal.execute(
                "frozen_transfer/source",
                {
                    "provenance": self.source.provenance,
                    "target": self.contract,
                },
                lambda: {
                    "knowledge": self.source.knowledge.to_dict(),
                    "provenance": self.source.provenance,
                    "target": self.contract,
                    "source_training_replayed": False,
                },
            )
            knowledge = KnowledgeState.from_dict(imported["knowledge"])
            if knowledge.snapshot_id != self.source.knowledge.snapshot_id:
                raise ValueError("Imported knowledge differs from the committed source")
            self.source.verify_unchanged()
            # This runs real multimodal retrieval/rewrite, Skill selection, tools,
            # recovery and verification. It does not read or fabricate `initial`.
            result = self.pipeline._deploy(tasks, knowledge)
            if knowledge.snapshot_id != self.source.knowledge.snapshot_id:
                raise RuntimeError("Transfer deployment mutated frozen knowledge")
            self.source.verify_unchanged()
            result = {
                **result,
                "transfer": {
                    **self.source.provenance,
                    **self.contract,
                    "source_read_only_verified": True,
                    "source_training_replayed": False,
                },
            }
            saved, _ = journal.execute(
                "frozen_transfer/results",
                {
                    "source_snapshot_id": knowledge.snapshot_id,
                    "target": self.contract,
                    "task_ids": [task.task_id for task in tasks],
                    "result": result,
                },
                lambda: result,
            )
            atomic_write_json(journal.root / "results/frozen_transfer.json", saved)
            return saved


def transfer_preflight(
    project,
    source_path,
    preparation,
    config_path,
    output,
    *,
    expected_snapshot_id=None,
    token_counter=None,
):
    source = read_frozen_source(source_path, expected_snapshot_id)
    validate_target_location(source, output)
    settings, datasets, binding, report = preflight(
        project, preparation, config_path, [source.dataset]
    )
    settings = replace(settings, experiment_name="cross_model_frozen_deployment")
    tasks = tuple(datasets[source.dataset]["deployment"])
    validate_target_tasks(source, tasks)
    contract = target_contract(project, settings, source, token_counter=token_counter)
    binding = {
        **binding,
        "mode": "cross_model_frozen_deployment",
        "settings": settings.to_dict(),
        "frozen_source": source.provenance,
        "frozen_transfer_target": contract,
    }
    report = {
        **report,
        "status": "preflight_passed_not_run",
        "source": source.provenance,
        "target": contract,
        "deployment_count": len(tasks),
        "accumulation_required": False,
        "source_read_only": True,
        "api_target_executor_supported": False,
    }
    return source, settings, datasets, binding, contract, report


def main(argv=None):
    import json
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--source-journal",
        type=Path,
        required=True,
        help="Source per-benchmark journal directory",
    )
    parser.add_argument("--expected-source-snapshot-id")
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, required=True, help="Target local v2 model configuration"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    source, settings, datasets, binding, contract, report = transfer_preflight(
        project,
        args.source_journal,
        args.preparation,
        args.config,
        args.output,
        expected_snapshot_id=args.expected_source_snapshot_id,
    )
    if args.report:
        atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Execute within a Slurm GPU allocation")
    if args.api_key_file:
        load_api_key_file(args.api_key_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Embedding credentials are missing; no inference started")
    source.verify_unchanged()
    bind_inference_resources(binding, datasets)
    from .runtime import ExperimentRuntime

    runtime = ExperimentRuntime(project, args.output, settings, binding)
    pipeline = runtime.dataset(source.dataset)
    result = FrozenTransferDeployment(source, pipeline, contract).deploy(
        datasets[source.dataset]["deployment"]
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
