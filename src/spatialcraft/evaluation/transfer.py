"""Cross-dataset transfer gains relative to target-domain vanilla."""

from __future__ import annotations

from dataclasses import dataclass

from spatialcraft.schemas import Trajectory

from .accuracy import accuracy


@dataclass(frozen=True, slots=True, kw_only=True)
class TransferMetrics:
    source_dataset: str
    target_dataset: str
    target_accuracy: float
    vanilla_target_accuracy: float
    absolute_gain: float
    relative_error_reduction: float


def transfer_metrics(
    transferred: tuple[Trajectory, ...],
    vanilla_target: tuple[Trajectory, ...],
    *,
    source_dataset: str,
    target_dataset: str,
) -> TransferMetrics:
    target = accuracy(transferred).accuracy
    vanilla = accuracy(vanilla_target).accuracy
    error = 1.0 - vanilla
    return TransferMetrics(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        target_accuracy=target,
        vanilla_target_accuracy=vanilla,
        absolute_gain=target - vanilla,
        relative_error_reduction=(target - vanilla) / error if error > 0 else 0.0,
    )


__all__ = ["TransferMetrics", "transfer_metrics"]
