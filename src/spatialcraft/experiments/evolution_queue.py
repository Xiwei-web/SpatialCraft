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
