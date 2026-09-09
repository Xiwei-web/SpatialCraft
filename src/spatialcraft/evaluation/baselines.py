"""Runnable, explicit configurations for the eight requested baselines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from spatialcraft.schemas import TaskSample, Trajectory


class BaselineName(str, Enum):
    COT = "CoT"
    TOOLS_ONLY_REACT = "Tools-only ReAct"
    RAG = "RAG"
    XSKILL = "XSkill"
    SKILL_PRO = "Skill-Pro"
    SMA = "SMA"
    MEMP = "MemP"
    MEMRL_R_GT = "MemRL-R/GT"


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineConfig:
    name: BaselineName
    tools: bool = False
    experience: bool = False
    experience_learning: bool = False
    skill: bool = False
    skill_learning: bool = False
    semantic_gradient: bool = False
    ppo_gate: bool = False
    visual_summary: bool = False
    variants: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


def standard_baselines() -> tuple[BaselineConfig, ...]:
    return (
        BaselineConfig(name=BaselineName.COT),
        BaselineConfig(name=BaselineName.TOOLS_ONLY_REACT, tools=True),
        BaselineConfig(name=BaselineName.RAG, experience=True),
        BaselineConfig(
            name=BaselineName.XSKILL,
            tools=True,
            experience=True,
            experience_learning=True,
            visual_summary=True,
        ),
        BaselineConfig(
            name=BaselineName.SKILL_PRO,
            tools=True,
            skill=True,
            skill_learning=True,
            semantic_gradient=True,
            ppo_gate=True,
        ),
        BaselineConfig(
            name=BaselineName.SMA,
            tools=True,
            experience=True,
            experience_learning=True,
            visual_summary=True,
            metadata={"memory": "experience_grounded_procedure"},
        ),
        BaselineConfig(
            name=BaselineName.MEMP,
            tools=True,
            experience=True,
            metadata={"memory": "procedural"},
        ),
        BaselineConfig(
            name=BaselineName.MEMRL_R_GT,
            tools=True,
            experience=True,
            experience_learning=True,
            variants=("R", "GT"),
        ),
    )


BaselineExecutor = Callable[
    [BaselineConfig, Sequence[TaskSample], int], tuple[Trajectory, ...]
]


class BaselineRunner:
    def __init__(self, executor: BaselineExecutor) -> None:
        self.executor = executor

    def run(
        self,
        tasks: Sequence[TaskSample],
        *,
        seed: int = 0,
        configs: tuple[BaselineConfig, ...] | None = None,
    ) -> dict[BaselineName, tuple[Trajectory, ...]]:
        selected = configs or standard_baselines()
        return {config.name: self.executor(config, tasks, seed) for config in selected}


__all__ = [
    "BaselineConfig",
    "BaselineName",
    "BaselineRunner",
    "standard_baselines",
]
