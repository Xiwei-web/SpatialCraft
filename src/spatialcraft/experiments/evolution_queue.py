"""Resumable per-version FIFO queues of distinct related training trajectories.

Queue entries reference immutable rollout commits, not copies of long histories.
The advantage is computed before selection from the original four-rollout task.
"""

from __future__ import annotations

import copy

from spatialcraft.knowledge.skill import SkillCreditAssigner
from spatialcraft.knowledge.skill.segments import skill_segments

from .journal import digest


class EvolutionQueue:
    def __init__(self, state=None):
        self.state = copy.deepcopy(
            state or {"schema_version": 1, "queues": {}, "seen_trajectory_ids": []}
        )
        if self.state.get("schema_version") != 1:
            raise ValueError("Unsupported evolution queue version")
        seen = self.state["seen_trajectory_ids"]
        if len(seen) != len(set(seen)):
            raise ValueError("Duplicate evolution queue trajectory IDs")
        for entries in self.state["queues"].values():
            ids = [e["trajectory_id"] for e in entries]
            if len(ids) != len(set(ids)) or not set(ids) <= set(seen):
                raise ValueError("Invalid per-parent queue membership")

    def enqueue(self, rows, *, task_index, active_references):
        if len(rows) != 4 or len({r.task.task_id for r in rows}) != 1:
            raise ValueError("Enqueue requires one complete four-rollout task")
        if len({r.trajectory_id for r in rows}) != 4:
            raise ValueError("Task rollout IDs must be unique")
        if any(r.trajectory_id in self.state["seen_trajectory_ids"] for r in rows):
            raise ValueError("Refusing to count a trajectory twice")
        if any(r.reward not in (0, 1) for r in rows):
            raise ValueError("Evolution requires verified binary rewards")
        advantages = {
            a.trajectory_id: a.advantage for a in SkillCreditAssigner().advantages(rows)
        }
        for row in rows:
            self.state["seen_trajectory_ids"].append(row.trajectory_id)
            # Multiple activation windows still count as one related trajectory.
            references = {
                s.skill_reference
                for s in skill_segments(row)
                if s.skill_reference is not None
            }
            if not references <= active_references:
                raise ValueError("A new rollout used an inactive Skill version")
            entry = {
                "trajectory_id": row.trajectory_id,
                "task_id": row.task.task_id,
                "source_key": f"tasks/{task_index:05d}/rollouts/{row.rollout_index:02d}/complete",
                "trajectory_sha256": digest(row.to_dict()),
                "advantage": advantages[row.trajectory_id],
                "sequence": task_index * 4 + row.rollout_index,
            }
            for reference in sorted(references):
                self.state["queues"].setdefault(reference, []).append(dict(entry))
        return advantages

    def take_round(self, *, batch_size=6, maximum_parents=2):
        if batch_size < 1 or maximum_parents < 1:
            raise ValueError("Evolution budgets must be positive")
        queues = self.state["queues"]
        eligible = sorted(
            (ref for ref, rows in queues.items() if len(rows) >= batch_size),
            key=lambda ref: (queues[ref][0]["sequence"], ref),
        )
        batches = {}
        for reference in eligible[:maximum_parents]:
            batches[reference] = queues[reference][:batch_size]
            queues[reference] = queues[reference][batch_size:]
        return batches

    def discard_inactive(self, active_references):
        discarded = {}
        for reference in list(self.state["queues"]):
            if reference not in active_references:
                entries = self.state["queues"].pop(reference)
                if entries:
                    discarded[reference] = entries
        return discarded


class RelatedEvolutionQueue:
    """Versioned semantic eligibility with reproducible 3-low/3-high batches."""

    def __init__(self, state=None):
        self.state = copy.deepcopy(
            state
            or {
                "schema_version": 2,
                "queues": {},
                "seen_trajectory_ids": [],
                "diagnosis_cache": {},
                "discovery_buckets": {},
            }
        )
        if self.state.get("schema_version") != 2:
            raise ValueError("Skill v2 cannot silently reuse v1 evolution queues")
        seen = self.state["seen_trajectory_ids"]
        if len(seen) != len(set(seen)):
            raise ValueError("Duplicate seen trajectories in semantic queue")
        for entries in self.state["queues"].values():
            ids = [entry["trajectory_id"] for entry in entries]
            if len(ids) != len(set(ids)) or not set(ids) <= set(seen):
                raise ValueError("Invalid semantic queue membership")
            if any(entry["reward"] not in (0, 1) for entry in entries):
                raise ValueError("Semantic queue requires binary task reward")
        self.state.setdefault("diagnosis_cache", {})
        self.state.setdefault("discovery_buckets", {})

    def enqueue(self, rows, *, task_index, diagnoses, active_references):
        if not rows or len({r.task.task_id for r in rows}) != 1:
            raise ValueError("Enqueue requires one complete task group")
        ids = [r.trajectory_id for r in rows]
        if len(ids) != len(set(ids)) or set(ids) & set(
            self.state["seen_trajectory_ids"]
        ):
            raise ValueError("Refusing to count a trajectory twice")
        if any(r.reward not in (0, 1) for r in rows):
            raise ValueError("Semantic evolution requires verified binary rewards")
        advantages = {
            a.trajectory_id: a.advantage for a in SkillCreditAssigner().advantages(rows)
        }
        for row in rows:
            self.state["seen_trajectory_ids"].append(row.trajectory_id)
            groups = {}
            for diagnosis in diagnoses.get(row.trajectory_id, ()):
                self.state["diagnosis_cache"][diagnosis["diagnosis_id"]] = diagnosis
                reference = diagnosis["skill_ref"]
                if reference is not None:
                    if reference not in active_references:
                        raise ValueError(
                            "Diagnosis refers to an inactive Skill version"
                        )
                    if not diagnosis["is_related"]:
                        continue
                    target = reference
                else:
                    if not diagnosis.get("needs_new_skill", False):
                        continue
                    key = diagnosis["discovery_key"]
                    target = "DISCOVERY:" + key
                    self.state["discovery_buckets"].setdefault(
                        key, diagnosis.get("discovery_need", key)
                    )
                groups.setdefault(target, []).append(diagnosis)
            for target, values in groups.items():
                self.state["queues"].setdefault(target, []).append(
                    {
                        "trajectory_id": row.trajectory_id,
                        "task_id": row.task.task_id,
                        "snapshot_id": row.knowledge_snapshot_id,
                        "source_key": f"tasks/{task_index:05d}/rollouts/{row.rollout_index:02d}/complete",
                        "trajectory_sha256": digest(row.to_dict()),
                        "reward": float(row.reward),
                        "advantage": advantages[row.trajectory_id],
                        "sequence": [task_index, row.rollout_index],
                        "diagnosis_ids": [v["diagnosis_id"] for v in values],
                        "transition_ids": list(
                            dict.fromkeys(
                                t for v in values for t in v["transition_ids"]
                            )
                        ),
                    }
                )
        return advantages

    def take_round(
        self,
        *,
        batch_size=6,
        preferred_low=3,
        preferred_high=3,
        maximum_parents=2,
        maximum_targets=2,
        maximum_discovery=1,
    ):
        if batch_size < 1 or preferred_low + preferred_high != batch_size:
            raise ValueError("Reward allocation must sum to the evolution batch size")
        if (
            min(
                preferred_low,
                preferred_high,
                maximum_parents,
                maximum_targets,
                maximum_discovery,
            )
            < 0
        ):
            raise ValueError("Negative evolution scheduling budget")
        queues = self.state["queues"]
        eligible = sorted(
            (k for k, v in queues.items() if len(v) >= batch_size),
            key=lambda k: (queues[k][0]["sequence"], k),
        )
        result = {}
        parents = discoveries = 0
        for target in eligible:
            discovery = target.startswith("DISCOVERY:")
            if len(result) >= maximum_targets:
                break
            if discovery and discoveries >= maximum_discovery:
                continue
            if not discovery and parents >= maximum_parents:
                continue
            rows = queues[target]
            lows = [e for e in rows if e["reward"] == 0][:preferred_low]
            highs = [e for e in rows if e["reward"] == 1][:preferred_high]
            selected = lows + highs
            selected_ids = {e["trajectory_id"] for e in selected}
            selected += [e for e in rows if e["trajectory_id"] not in selected_ids][
                : batch_size - len(selected)
            ]
            selected.sort(key=lambda e: e["sequence"])
            selected_ids = {e["trajectory_id"] for e in selected}
            queues[target] = [e for e in rows if e["trajectory_id"] not in selected_ids]
            result[target] = selected
            discoveries += int(discovery)
            parents += int(not discovery)
        return result

    def discard_inactive(self, active_references):
        removed = {}
        for target in list(self.state["queues"]):
            if not target.startswith("DISCOVERY:") and target not in active_references:
                removed[target] = self.state["queues"].pop(target)
        return removed
