"""Task-level Experience barriers plus per-parent trajectory-queue evolution.

The old pipelines/accumulation.py is a generic component; this strict protocol
requires all builders explicitly and makes each knowledge commit crash-safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from spatialcraft.agent.decision import is_controlled_failure
from spatialcraft.knowledge.experience import ExperienceBank
from spatialcraft.knowledge.skill import SeedCatalog, SkillPool
from spatialcraft.schemas import (
    RetrievedExperienceRef,
    TaskSample,
    TaskSplit,
    Trajectory,
    TrajectoryStatus,
)
from spatialcraft.storage.atomic_io import canonical_json_bytes, file_lock

from .journal import RunJournal, digest
from .settings import ExperimentSettings


@dataclass(frozen=True, slots=True)
class KnowledgeState:
    experiences: ExperienceBank
    skills: SkillPool

    def __post_init__(self):
        if not self.experiences.frozen or not self.skills.frozen:
            raise ValueError("Task knowledge must be frozen")

    @property
    def snapshot_id(self):
        return "knowledge_" + digest(self.to_dict())

    def to_dict(self):
        return {
            "experience_bank": self.experiences.to_dict(),
            "skill_pool": self.skills.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            ExperienceBank.from_dict(data["experience_bank"], frozen=True),
            SkillPool.from_dict(data["skill_pool"], frozen=True),
        )

    def copy(self):
        # Schema roundtrip also isolates nested metadata containers per rollout.
        return self.from_dict(json.loads(canonical_json_bytes(self.to_dict())))


class ProtocolPipeline:
    def __init__(
        self,
        settings: ExperimentSettings,
        journal: RunJournal,
        *,
        prepare_experiences,
        rollout,
        update_experiences,
        evolve_skills,
        artifact_validator=None,
    ):
        self.settings, self.journal = settings, journal
        self.prepare_experiences, self.rollout = prepare_experiences, rollout
        self.update_experiences, self.evolve_skills = update_experiences, evolve_skills
        self.artifact_validator = artifact_validator
        if any(
            fn is None
            for fn in (prepare_experiences, rollout, update_experiences, evolve_skills)
        ):
            raise ValueError(
                "Full SpatialCraft requires all knowledge/rollout callbacks"
            )

    def _prepare(self, task, knowledge, phase):
        owner = getattr(self.prepare_experiences, "__self__", None)
        if hasattr(owner, "set_scope"):
            owner.set_scope(
                phase=phase, task_id=task.task_id, snapshot_id=knowledge.snapshot_id
            )
        value = self.prepare_experiences(task.without_reference_answer(), knowledge)
        if isinstance(value, dict):
            return value
        return {"experiences": [r.to_dict() for r in value]}

    def _update(self, knowledge, rows, prepared):
        if self.settings.is_v2:
            return self.update_experiences(
                knowledge, rows, retrieval_audit=prepared.get("retrieval_audit", {})
            )
        return self.update_experiences(knowledge, rows)

    def _export(self, relative, value):
        from spatialcraft.storage.atomic_io import canonical_json_bytes

        from .prepare import _immutable

        _immutable(
            self.journal.root / relative, canonical_json_bytes(value), private=True
        )

    def _validate(self, trajectory, task, frozen, index, seed):
        if (
            trajectory.task.to_dict() != task.to_dict()
            or trajectory.rollout_index != index
            or trajectory.random_seed != seed
            or trajectory.knowledge_snapshot_id != frozen.snapshot_id
        ):
            raise ValueError("Rollout task/index/seed/frozen snapshot binding mismatch")
        allowed_status = trajectory.status in {
            TrajectoryStatus.COMPLETED,
            TrajectoryStatus.TRUNCATED,
        } or (
            trajectory.status is TrajectoryStatus.FAILED
            and is_controlled_failure(trajectory)
        )
        if trajectory.reward not in (0.0, 1.0) or not allowed_status:
            raise ValueError(
                "Require completed/truncated trajectories with binary rewards; infrastructure errors must propagate"
            )
        if self.artifact_validator is not None:
            self.artifact_validator(trajectory)

    def accumulate(self, tasks: tuple[TaskSample, ...]) -> KnowledgeState:
        if (
            not tasks
            or len({t.dataset for t in tasks}) != 1
            or len({t.task_id for t in tasks}) != len(tasks)
        ):
            raise ValueError("Expected unique training tasks from one benchmark")
        if any(t.split is not TaskSplit.TRAIN for t in tasks):
            raise ValueError("Never accumulate deployment/test samples")
        # Top-level lock prevents two processes executing different future tasks.
        with file_lock(self.journal.root / ".pipeline.lock", timeout=0):
            initial, _ = self.journal.execute(
                "initial",
                {
                    "settings": self.settings.to_dict(),
                    "task_ids": [t.task_id for t in tasks],
                },
                lambda: {
                    **KnowledgeState(
                        ExperienceBank().freeze(),
                        (
                            SkillPool()
                            if self.settings.is_v2
                            and self.settings.ablations.get("no_skill")
                            else SeedCatalog.pool()
                        ).freeze(),
                    ).to_dict(),
                    "dataset": tasks[0].dataset,
                    "training_task_ids": [t.task_id for t in tasks],
                },
            )
            current = KnowledgeState.from_dict(initial)
            evolution_state = None
            for task_index, task in enumerate(tasks):
                prefix = f"tasks/{task_index:05d}"
                frozen, before = current.copy(), current.snapshot_id
                prepared, _ = self.journal.execute(
                    prefix + "/retrieve_rewrite",
                    {
                        "task": task.without_reference_answer().to_dict(),
                        "snapshot": before,
                    },
                    lambda task=task, frozen=frozen: self._prepare(
                        task, frozen, "accumulation_execution"
                    ),
                )
                if frozen.snapshot_id != before:
                    raise RuntimeError("Retrieval modified frozen knowledge")
                refs = tuple(
                    RetrievedExperienceRef.from_dict(r) for r in prepared["experiences"]
                )
                rows = []
                for index in range(self.settings.rollouts_per_task):
                    seed = (
                        self.settings.seed
                        + task_index * self.settings.rollouts_per_task
                        + index
                    )
                    base = frozen.copy()
                    value, _ = self.journal.execute(
                        prefix + f"/rollouts/{index:02d}/complete",
                        {
                            "task": task.to_dict(),
                            "snapshot": before,
                            "retrieved": prepared,
                            "seed": seed,
                        },
                        lambda task=task, base=base, refs=refs, index=index, seed=seed, prefix=prefix: (
                            self.rollout(
                                task,
                                base,
                                refs,
                                index,
                                seed,
                                prefix + f"/rollouts/{index:02d}/execution",
                                False,
                            ).to_dict()
                        ),
                    )
                    row = Trajectory.from_dict(value)
                    if base.snapshot_id != before:
                        raise RuntimeError("A rollout modified its frozen knowledge")
                    self._validate(row, task, frozen, index, seed)
                    rows.append(row)
                    self._export(
                        f"rollouts/training/{task_index:05d}/{index:02d}.json", value
                    )
                updated, _ = self.journal.execute(
                    prefix + "/experience_update",
                    {"snapshot": before, "trajectories": [t.to_dict() for t in rows]},
                    lambda frozen=frozen, rows=tuple(rows), prepared=prepared: (
                        self._update(frozen.copy(), rows, prepared)
                    ),
                )
                bank = ExperienceBank.from_dict(updated["experience_bank"], frozen=True)
                if len(bank.active()) > self.settings.experience_capacity:
                    raise ValueError("Experience capacity exceeded")
                current = KnowledgeState(bank, frozen.skills)
                self._export(f"experiences/task-{task_index:05d}.json", updated)
                self._export(
                    f"checkpoints/task-{task_index:05d}.json", current.to_dict()
                )
                # One scheduling round per completed task, never inside its four
                # independent rollouts. The builder consumes per-parent batches.
                evolved, _ = self.journal.execute(
                    f"evolution/{task_index:05d}/skill_evolution",
                    {
                        "knowledge": current.to_dict(),
                        "trajectories": [r.to_dict() for r in rows],
                        "evolution_state": evolution_state,
                    },
                    lambda current=current, rows=tuple(rows), task_index=task_index, pending=evolution_state: (
                        self.evolve_skills(current.copy(), rows, task_index, pending)
                    ),
                )
                pool = SkillPool.from_dict(evolved["skill_pool"], frozen=True)
                if len(pool.active()) > self.settings.skill_capacity:
                    raise ValueError("Skill capacity exceeded")
                evolution_state = evolved["evolution_state"]
                current = KnowledgeState(current.experiences, pool)
                self._export(f"skills/round-{task_index:05d}.json", evolved)
                self._export(
                    f"checkpoints/round-{task_index:05d}.json",
                    {**current.to_dict(), "evolution_state": evolution_state},
                )
            self._export("checkpoints/pending_evolution.json", evolution_state)
            final, _ = self.journal.execute(
                "frozen_deployment_snapshot",
                {"snapshot_id": current.snapshot_id},
                current.to_dict,
            )
            self._export("checkpoints/frozen_deployment.json", final)
            return KnowledgeState.from_dict(final)

    def deploy(self, tasks: tuple[TaskSample, ...], knowledge: KnowledgeState) -> dict:
        with file_lock(self.journal.root / ".pipeline.lock", timeout=0):
            initial = self.journal.read_committed("initial")
            committed = KnowledgeState.from_dict(
                self.journal.read_committed("frozen_deployment_snapshot")
            )
            if committed.snapshot_id != knowledge.snapshot_id:
                raise ValueError("Deployment must use the committed final snapshot")
            if any(t.dataset != initial["dataset"] for t in tasks):
                raise ValueError("Deployment benchmark differs from training")
            if set(initial["training_task_ids"]) & {t.task_id for t in tasks}:
                raise ValueError("Training/deployment task IDs overlap")
            return self._deploy(tasks, knowledge)

    def _deploy(self, tasks: tuple[TaskSample, ...], knowledge: KnowledgeState) -> dict:
        from .protocol import split_category

        if not tasks or any(t.split is not TaskSplit.TEST for t in tasks):
            raise ValueError("Deployment requires nonempty test split")
        if (
            len({t.task_id for t in tasks}) != len(tasks)
            or len({t.dataset for t in tasks}) != 1
        ):
            raise ValueError("Deployment tasks must be unique within one benchmark")
        for task in tasks:
            split_category(
                task
            )  # Fail before inference if metric categories are missing.
        before = knowledge.snapshot_id
        rows = []
        for index, task in enumerate(tasks):
            prefix = f"deployment/{index:05d}"
            prepared, _ = self.journal.execute(
                prefix + "/retrieve_rewrite",
                {"task": task.without_reference_answer().to_dict(), "snapshot": before},
                lambda task=task: self._prepare(task, knowledge.copy(), "deployment"),
            )
            refs = tuple(
                RetrievedExperienceRef.from_dict(r) for r in prepared["experiences"]
            )
            seed = self.settings.seed + index
            base = knowledge.copy()
            value, _ = self.journal.execute(
                prefix + "/complete",
                {
                    "task": task.to_dict(),
                    "snapshot": before,
                    "retrieved": prepared,
                    "seed": seed,
                },
                lambda task=task, base=base, refs=refs, seed=seed, prefix=prefix: (
                    self.rollout(
                        task, base, refs, 0, seed, prefix + "/execution", True
                    ).to_dict()
                ),
            )
            row = Trajectory.from_dict(value)
            self._validate(row, task, knowledge, 0, seed)
            if base.snapshot_id != before or knowledge.snapshot_id != before:
                raise RuntimeError("Deployment mutated knowledge")
            rows.append(row)
            self._export(f"rollouts/deployment/{index:05d}.json", value)
        from collections import defaultdict

        categories = defaultdict(list)
        for row in rows:
            categories[split_category(row.task)].append(row.reward)
        result = {
            "snapshot_id": before,
            "count": len(rows),
            "correct": sum(t.reward for t in rows),
            "accuracy": sum(t.reward for t in rows) / len(rows),
            "by_category": {
                key: {"count": len(v), "accuracy": sum(v) / len(v)}
                for key, v in categories.items()
            },
            "knowledge_updated": False,
            "deployment_rollouts_per_task": 1,
        }
        if self.settings.is_v2:
            from spatialcraft.evaluation.protocol_metrics_v2 import protocol_metrics

            result["protocol_metrics_path"] = "results/protocol_metrics.json"
            self._export(
                result["protocol_metrics_path"],
                protocol_metrics(
                    rows,
                    knowledge,
                    token_counter=getattr(self, "metric_token_counter", None),
                    tokenizer_id=getattr(self, "metric_tokenizer_id", None),
                ),
            )
        self._export("results/deployment.json", result)
        if self.settings.is_v2:
            from spatialcraft.storage.atomic_io import atomic_write_json

            from .usage import cost_report

            atomic_write_json(
                self.journal.root / "results/usage.json",
                cost_report(self.journal.root, len(rows)),
            )
        return result
