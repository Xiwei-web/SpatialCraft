"""Load the fixed initial procedural-skill catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from spatialcraft.schemas import SkillEvolutionType, SkillItem

from .pool import SkillPool

DEFAULT_SEED_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "skills" / "seed_skills.yaml"
)


class SeedCatalog:
    @staticmethod
    def load(path: str | Path = DEFAULT_SEED_PATH) -> tuple[SkillItem, ...]:
        payload: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported seed-skill schema version")
        rows = payload.get("skills")
        if not isinstance(rows, list) or not rows:
            raise ValueError("seed catalog must contain a non-empty skills list")
        skills = tuple(
            SkillItem(
                skill_id=str(row["skill_id"]),
                name=str(row["name"]),
                initiation=str(row["initiation"]),
                policy=tuple(str(step) for step in row["policy"]),
                termination=str(row["termination"]),
                evolution_type=SkillEvolutionType.SEED,
                metadata=dict(row.get("metadata") or {}),
            )
            for row in rows
        )
        if len({item.name for item in skills}) != len(skills):
            raise ValueError("seed skill names must be unique")
        return skills

    @classmethod
    def pool(cls, path: str | Path = DEFAULT_SEED_PATH) -> SkillPool:
        return SkillPool(cls.load(path))


__all__ = ["DEFAULT_SEED_PATH", "SeedCatalog"]
