"""Sequential MemRL environment learning followed by frozen spatial deployment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import isfinite
from numbers import Real
from pathlib import Path

from memp.runner import (
    _task_key,
    _totals,
    _trajectory_usage,
    embedding_config,
    recovery_profiles,
    semantic_trajectory,
)
from memp.runner import (
    model_config as _shared_model_config,
)
from spatialcraft.agent import (
    ActionParser,
    ExecutionConfig,
    ExecutionLoop,
    StateBuilder,
)
from spatialcraft.agent.decision import BUDGET_POLICY
from spatialcraft.experiments.journal import RunJournal, digest
from spatialcraft.experiments.protocol import public_task
from spatialcraft.experiments.react_api import (
    SYSTEM,
    ReActResponsesProvider,
    ScopedToolExecutor,
    TaskMedia,
    ToolOnlyComposer,
)
from spatialcraft.experiments.rollout import JournaledRollout
from spatialcraft.experiments.run_react_baseline import (
    PrivateReward,
    report_for,
    trajectory_row,
)
from spatialcraft.models import MessageRole, ModelMessage
from spatialcraft.models.providers.openai_embeddings import OpenAIEmbeddingsProvider
from spatialcraft.schemas import TaskSplit
from spatialcraft.storage import StorageLayout
from spatialcraft.storage.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    file_lock,
)
from spatialcraft.tools import ArtifactStore
from spatialcraft.tools.real import create_real_tool_registry

from .memory import (
    MemoryIndex,
    MemoryRecord,
    calibrate_threshold,
    create_memory,
    render_memory_prompt,
    update_utilities,
)

POLICY = "memrl_sma_spatial_v1"


@dataclass(frozen=True)
class Settings:
    model: str = "gpt-5.4-mini"
    reasoning_effort: str = "medium"
    max_output_tokens: int = 16384
    temperature: float = 0.0
    environment_rollouts: int = 1
    environment_passes: int = 1
    max_steps: int = 50
    top_k: int = 3
    candidate_k: int = 10
    utility_weight: float = 0.5
    learning_rate: float = 0.3
    q_init: float = 0.0
    similarity_threshold: float | None = None
    threshold_quantile: float = 0.8
    embedding_model: str = "text-embedding-3-large"

    def __post_init__(self):
        if self.model not in {"gpt-5.4", "gpt-5.4-mini"}:
            raise ValueError("Unsupported actor model")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported reasoning effort")
        if self.embedding_model not in {
            "text-embedding-3-large",
            "text-embedding-3-small",
        }:
            raise ValueError("Unsupported embedding model")
        for name in (
            "max_output_tokens",
            "environment_rollouts",
            "environment_passes",
            "max_steps",
            "top_k",
            "candidate_k",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.top_k > self.candidate_k:
            raise ValueError("top_k must be <= candidate_k")
        for name, low, high in (
            ("temperature", 0, 2),
            ("utility_weight", 0, 1),
            ("learning_rate", 0, 1),
            ("q_init", 0, 1),
            ("threshold_quantile", 0, 1),
            ("similarity_threshold", -1, 1),
        ):
            value = getattr(self, name)
            if name == "similarity_threshold" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def model_config(project, settings):
    config = _shared_model_config(project, settings)
    return replace(config, metadata={**config.metadata, "purpose": POLICY})


class MemRLComposer(ToolOnlyComposer):
    def __init__(self, model, tools, *, media, hits=(), snapshot_id, variant):
        super().__init__(model, tools, media=media)
        self.hits, self.snapshot_id, self.variant = tuple(hits), snapshot_id, variant

    def compose(self, task, state):
        request = super().compose(task, state)
        messages = list(request.messages)
        if self.hits:
            position = next(
                i
                for i, message in enumerate(messages)
                if message.role is MessageRole.USER
            )
            messages.insert(
                position,
                ModelMessage.text(
                    MessageRole.DEVELOPER, render_memory_prompt(self.hits)
                ),
            )
        return replace(
            request,
            messages=tuple(messages),
            metadata={
                **request.metadata,
                "baseline_policy": POLICY,
                "memrl_variant": self.variant,
                "snapshot_id": self.snapshot_id,
                "memory_ids": [hit.record.memory_id for hit in self.hits],
                "retrieval_scores": [hit.score for hit in self.hits],
                "recovery_profiles": recovery_profiles(self.model),
            },
        )


def last_model_output(trajectory, journal, prefix):
    """Retain even an unparseable last response; omit raw API and verifier data."""
    result = {"final_answer": trajectory.final_answer}
    events = trajectory.metadata.get("generation_events", ())
    if events:
        event = events[-1]
        suffix = {
            "normal": "model",
            "recovery": "recovery/model",
            "forced_final": "forced_final/model",
        }[event["kind"]]
        wire = journal.read_committed(
            f"{prefix}/steps/{event['step_index']:04d}/{suffix}"
        )
        result["text"] = wire.get("text")
        result["tool_calls"] = [
            {"name": item["name"], "arguments": item["arguments"]}
            for item in wire.get("tool_calls", ())
        ]
    return result


class DatasetRunner:
    """Variant-isolated durable learning with an append-only update event chain."""

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
        registry=None,
        provider_factory=None,
    ):
        if variant not in {"R", "GT"} or builder.variant != variant:
            raise ValueError("Runner and reflection builder must share R or GT variant")
        if builder.config != config:
            raise ValueError(
                "Actor and reflection must use identical model configuration"
            )
        self.output, self.name, self.identity = Path(output), name, identity
        self.settings, self.config, self.variant = settings, config, variant
        self.embedder, self.builder = embedder, builder
        self.registry = (
            registry if registry is not None else create_real_tool_registry()
        )
        self.provider_factory = provider_factory
        self.root = self.output / name
        self.layout = StorageLayout(self.root / "tool_store")
        self.store = ArtifactStore(self.layout, name)
        self.journal = RunJournal(
            self.root,
            {
                **resources,
                "policy": POLICY,
                "variant": variant,
                "dataset": name,
                "data_sha256": digest(identity),
                "settings": asdict(settings),
                "model_id": config.model_id,
                "generation": asdict(config.generation),
                "endpoint": config.api.resolved_base_url(),
                "actor_system": SYSTEM,
                "tool_schemas": [asdict(t) for t in self.registry.definitions()],
                "embedding_identity": embedder.identity,
                "protocol": "sequential_environment_learning_frozen_deployment",
                "budget": {"policy": BUDGET_POLICY, **recovery_profiles(config)},
                "retrieval": "item_candidates_then_candidate_zscore_similarity_and_Q",
                "self_exclusion": "same_task_id_or_exact_public_content_in_all_phases",
                "historical_media": "semantic_trace_JSON_and_media_descriptors",
                "checkpoint": "last_environment_pass_no_heldout_selection",
            },
        )

    def _rollout(self, task, reference, *, prefix, hits, snapshot, rollout_index=0):
        media = TaskMedia(task, self.identity, self.layout)
        provider = (
            self.provider_factory(self.config, media)
            if self.provider_factory
            else ReActResponsesProvider(self.config, media)
        )
        loop = ExecutionLoop(
            provider=provider,
            composer=MemRLComposer(
                self.config,
                self.registry,
                media=media,
                hits=hits,
                snapshot_id=snapshot,
                variant=self.variant,
            ),
            action_parser=ActionParser(self.registry),
            tool_executor=ScopedToolExecutor(self.registry, self.store, media),
            reward=PrivateReward(reference),
            skill_controller=None,
            config=ExecutionConfig(
                max_steps=self.settings.max_steps, knowledge_snapshot_id=snapshot
            ),
        )
        try:
            trajectory = JournaledRollout(loop, self.journal).run(
                task,
                prefix=prefix,
                initial_state=StateBuilder().initial(task),
                rollout_index=rollout_index,
                random_seed=None,
            )
        finally:
            if (
                self.provider_factory is None
                and getattr(provider, "_client", None) is not None
            ):
                provider._client.close()
        row = {
            **trajectory_row(task, trajectory, self.journal, prefix),
            "usage": _trajectory_usage(trajectory, self.journal, prefix),
            "rollout_index": rollout_index,
            "variant": self.variant,
            "memory_ids": [hit.record.memory_id for hit in hits],
            "memory_snapshot": snapshot,
        }
        return trajectory, row

    def _validate_pairs(self, public, private, phase):
        if (
            not public
            or len(public) != len(private)
            or len({t.task_id for t in public}) != len(public)
        ):
            raise ValueError("Empty, duplicated, or misaligned split")
        expected = TaskSplit.TRAIN if phase == "environment" else TaskSplit.TEST
        for task, reference in zip(public, private, strict=True):
            if digest(task.to_dict()) != digest(public_task(reference)):
                raise ValueError(
                    "Public/private pairing mismatch or private labels in actor task"
                )
            if task.dataset != self.name or task.split != expected:
                raise ValueError("Dataset or environment/deployment split mismatch")
            if task.metadata.get("experiment_split", phase) != phase:
                raise ValueError("Experiment phase mismatch")
            if reference.reference_answer is None:
                raise ValueError(
                    "Environment and deployment require private scoring labels"
                )

    def _retrieval(self, prefix, task, index, vector, snapshot, threshold):
        selection = index.retrieve(
            vector,
            dataset=self.name,
            task_id=task.task_id,
            content_key=_task_key(task, self.identity),
            threshold=threshold,
            candidate_k=self.settings.candidate_k,
            top_k=self.settings.top_k,
            utility_weight=self.settings.utility_weight,
        )
        committed, _ = self.journal.execute(
            prefix + "/retrieval",
            {
                "snapshot": snapshot,
                "public_task": public_task(task),
                "query_vector_sha256": digest(vector),
            },
            lambda: selection.audit,
        )
        if committed != selection.audit:
            raise ValueError("Replayed retrieval disagrees with committed memory state")
        return selection.hits

    def environment(self, public, private):
        self._validate_pairs(public, private, "environment")
        # Calibration sees public environment queries only, before memory writing.
        # One commit per query preserves partial progress across large datasets.
        task_vectors = []
        for i, task in enumerate(public):
            embedded, _ = self.journal.execute(
                f"calibration/{i:05d}/embedding",
                {
                    "query": task.question,
                    "identity": self.embedder.identity,
                },
                lambda task=task: {"vector": self.embedder.embed((task.question,))[0]},
            )
            task_vectors.append(embedded["vector"])
        calibration, _ = self.journal.execute(
            "calibration/threshold",
            {
                "public_task_ids": [task.task_id for task in public],
                "vectors_sha256": digest(task_vectors),
                "explicit_threshold": self.settings.similarity_threshold,
                "quantile": self.settings.threshold_quantile,
            },
            lambda: {
                "threshold": (
                    self.settings.similarity_threshold
                    if self.settings.similarity_threshold is not None
                    else calibrate_threshold(
                        task_vectors, self.settings.threshold_quantile
                    )
                ),
                "source": "explicit"
                if self.settings.similarity_threshold is not None
                else "environment_public_pairwise_cosine_quantile",
                "task_count": len(public),
                "pair_count": len(public) * (len(public) - 1) // 2,
                "quantile": self.settings.threshold_quantile,
                "single_task_fallback": len(public) == 1
                and self.settings.similarity_threshold is None,
            },
        )
        records, vectors, rows, reflection_responses = [], [], [], []
        index = MemoryIndex((), ())
        head = digest(
            {
                "empty_bank": True,
                "binding": self.journal.binding_digest,
                "calibration": calibration,
            }
        )
        total = (
            len(public)
            * self.settings.environment_rollouts
            * self.settings.environment_passes
        )
        for pass_index in range(self.settings.environment_passes):
            for i, (task, reference) in enumerate(zip(public, private, strict=True)):
                for k in range(self.settings.environment_rollouts):
                    prefix = f"environment/{pass_index:03d}/{i:05d}/{k:03d}"
                    vector = task_vectors[i]
                    hits = self._retrieval(
                        prefix, task, index, vector, head, calibration["threshold"]
                    )
                    trajectory, row = self._rollout(
                        task,
                        reference,
                        prefix=prefix,
                        hits=hits,
                        snapshot=head,
                        rollout_index=k,
                    )
                    # PrivateReward is binary. Controlled invalid-action failures
                    # receive zero; transport/tool infrastructure failures propagate.
                    reward = float(row["reward"])
                    selected_ids = [hit.record.memory_id for hit in hits]
                    updated = update_utilities(
                        records, selected_ids, reward, self.settings.learning_rate
                    )
                    reflection_key = prefix + "/reflection"
                    reflection = self.builder.build(
                        self.journal,
                        reflection_key,
                        task=task,
                        reference=reference if self.variant == "GT" else None,
                        trace=semantic_trajectory(trajectory, self.layout),
                        model_output=last_model_output(
                            trajectory, self.journal, prefix
                        ),
                        reward=reward,
                        media=TaskMedia(task, self.identity, self.layout),
                        phase="environment",
                    )
                    reflection_responses.append(
                        self.journal.read_committed(reflection_key)
                    )
                    record = create_memory(
                        dataset=self.name,
                        task_id=task.task_id,
                        question=task.question,
                        choices=task.choices,
                        media_hashes=[
                            self.identity["media"][im.uri] for im in task.images
                        ],
                        reflection=reflection,
                        source_key=prefix,
                        q_init=self.settings.q_init,
                    )
                    # Commit a small complete transition of the bank, not an
                    # in-place mutation or quadratic-sized full bank per task.
                    selected_set = set(selected_ids)
                    event = {
                        "parent": head,
                        "reward": reward,
                        "selected_ids": selected_ids,
                        "updated_utilities": [
                            {
                                "memory_id": item.memory_id,
                                "q_value": item.q_value,
                                "visits": item.visits,
                            }
                            for item in updated
                            if item.memory_id in selected_set
                        ],
                        "new_record": record.to_dict(),
                        "new_vector": vector,
                        "trajectory_key": prefix + "/trajectory",
                        "trajectory_sha256": digest(
                            self.journal.read_committed(prefix + "/trajectory")
                        ),
                        "reflection_key": reflection_key,
                    }
                    committed, _ = self.journal.execute(
                        prefix + "/memory_update",
                        {
                            "event": event,
                        },
                        lambda event=event: {**event, "head": digest(event)},
                    )
                    if committed != {**event, "head": digest(event)}:
                        raise ValueError("Memory update commit mismatch")
                    records = [*updated, record]
                    vectors.append(vector)
                    index = index.advance(records, vector)
                    head = committed["head"]
                    row.update(
                        {
                            "pass_index": pass_index,
                            "new_memory_id": record.memory_id,
                            "memory_after": head,
                        }
                    )
                    rows.append(row)
                    atomic_write_json(self.root / f"predictions/{prefix}.json", row)
                    self._progress("environment", rows, total)
        complete, _ = self.journal.execute(
            "environment_complete",
            {
                "task_ids": [t.task_id for t in public],
                "rows": rows,
                "head": head,
            },
            lambda: {"rows": rows, "head": head},
        )
        value = {
            "policy": POLICY,
            "variant": self.variant,
            "settings": asdict(self.settings),
            "records": [record.to_dict() for record in records],
            "vectors": vectors,
            "calibration": calibration,
            "threshold": calibration["threshold"],
            "head": head,
            "environment_sha256": digest(complete),
        }
        frozen, _ = self.journal.execute(
            "frozen_memory",
            {"bank_sha256": digest(value)},
            lambda: {**value, "snapshot_id": digest(value)},
        )
        atomic_write_json(self.root / "memory/frozen.json", frozen)
        report = self._report("environment", rows, total)
        report.update(
            {
                "memory_count": len(records),
                "memory_snapshot": frozen["snapshot_id"],
                "reflection_calls": len(reflection_responses),
                "reflection_usage": _totals(reflection_responses),
                "utility_updates": sum(record.visits for record in records),
                "calibration": calibration,
                "reflection_call_semantics": "committed logical calls; cached replay generates no new calls",
            }
        )
        atomic_write_json(self.root / "results/environment.json", report)
        return report

    def deployment(self, public, private):
        self._validate_pairs(public, private, "deployment")
        frozen = self.journal.read_committed("frozen_memory")
        value = {key: val for key, val in frozen.items() if key != "snapshot_id"}
        if digest(value) != frozen["snapshot_id"]:
            raise ValueError("Frozen memory digest mismatch")
        if frozen["variant"] != self.variant or frozen["settings"] != asdict(
            self.settings
        ):
            raise ValueError("Frozen memory variant/settings mismatch")
        records = [MemoryRecord.from_dict(item) for item in frozen["records"]]
        index = MemoryIndex(records, frozen["vectors"])
        rows = []
        for i, (task, reference) in enumerate(zip(public, private, strict=True)):
            prefix = f"deployment/{i:05d}"
            embedded, _ = self.journal.execute(
                prefix + "/query_embedding",
                {
                    "query": task.question,
                    "identity": self.embedder.identity,
                },
                lambda task=task: {"vector": self.embedder.embed((task.question,))[0]},
            )
            hits = self._retrieval(
                prefix,
                task,
                index,
                embedded["vector"],
                frozen["snapshot_id"],
                frozen["threshold"],
            )
            _, row = self._rollout(
                task,
                reference,
                prefix=prefix,
                hits=hits,
                snapshot=frozen["snapshot_id"],
            )
            rows.append(row)
            atomic_write_json(self.root / f"predictions/{prefix}.json", row)
            self._progress("deployment", rows, len(public))
        if self.journal.read_committed("frozen_memory") != frozen:
            raise ValueError("Memory changed during frozen deployment")
        report = self._report("deployment", rows, len(public))
        report.update(
            {
                "memory_count": len(records),
                "memory_snapshot": frozen["snapshot_id"],
                "threshold": frozen["threshold"],
            }
        )
        atomic_write_json(self.root / "results/deployment.json", report)
        atomic_write_bytes(
            self.root / "results/predictions.jsonl",
            b"".join(canonical_json_bytes(row) for row in rows),
            mode=0o600,
        )
        return report

    def _report(self, phase, rows, total):
        return {
            **report_for(self.name, rows, total),
            "policy": POLICY,
            "variant": self.variant,
            "phase": phase,
            "settings": asdict(self.settings),
            "usage": _totals(rows),
        }

    def _progress(self, phase, rows, total):
        report = self._report(phase, rows, total)
        atomic_write_json(self.root / f"progress/{phase}.json", report)
        print(
            json.dumps(
                {
                    "dataset": self.name,
                    "variant": self.variant,
                    "phase": phase,
                    "completed": len(rows),
                    "total": total,
                    "correct": report["correct"],
                }
            ),
            flush=True,
        )

    def run(self, stage, environment, deployment):
        if stage not in {"all", "environment", "deployment"}:
            raise ValueError(f"Unknown MemRL stage: {stage}")
        # Validate the entire protocol before issuing a first external call.
        self._validate_pairs(*environment, "environment")
        self._validate_pairs(*deployment, "deployment")
        if {t.task_id for t in environment[0]} & {t.task_id for t in deployment[0]}:
            raise ValueError("Environment and deployment task IDs overlap")
        with file_lock(self.root / ".memrl.lock", timeout=0):
            if stage in {"all", "environment"}:
                result = self.environment(*environment)
            if stage in {"all", "deployment"}:
                result = self.deployment(*deployment)
        return {**result, "stage": stage, "status": "completed"}


__all__ = [
    "POLICY",
    "DatasetRunner",
    "MemRLComposer",
    "OpenAIEmbeddingsProvider",
    "Settings",
    "embedding_config",
    "last_model_output",
    "model_config",
]
