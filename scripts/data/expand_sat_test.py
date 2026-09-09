"""Circular-expand the official SAT test split from 150 to 300 rows."""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_SOURCE_ROWS = 150
EXPECTED_ANSWER_COUNT = 2


def _source_path(test_path: Path, backup_path: Path) -> Path:
    """Create the immutable source backup once and use it on subsequent runs."""

    if backup_path.exists():
        return backup_path
    source = pq.read_table(test_path)
    if source.num_rows != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_ROWS} official test rows, "
            f"found {source.num_rows} in {test_path}"
        )
    shutil.copy2(test_path, backup_path)
    return backup_path


def expand(test_path: Path, backup_path: Path) -> None:
    """Write two answer-order rotations for every official SAT test question."""

    source_path = _source_path(test_path, backup_path)
    source = pq.read_table(source_path)
    rows = source.to_pylist()
    if len(rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"source backup must contain {EXPECTED_SOURCE_ROWS} rows, found {len(rows)}"
        )

    expanded: list[dict[str, object]] = []
    for source_index, row in enumerate(rows):
        answers = list(row["answers"])
        correct_answer = row["correct_answer"]
        if len(answers) != EXPECTED_ANSWER_COUNT:
            raise ValueError(
                f"row {source_index} has {len(answers)} answers; expected 2"
            )
        if answers.count(correct_answer) != 1:
            raise ValueError(
                f"row {source_index} must contain its correct answer exactly once"
            )
        correct_first = [correct_answer, *(a for a in answers if a != correct_answer)]
        for circular_shift in range(EXPECTED_ANSWER_COUNT):
            rotated = correct_first[circular_shift:] + correct_first[:circular_shift]
            output = dict(row)
            output.update(
                {
                    "answers": rotated,
                    "source_row_index": source_index,
                    "circular_shift": circular_shift,
                    "correct_answer_index": rotated.index(correct_answer),
                    "correct_answer_label": "AB"[rotated.index(correct_answer)],
                }
            )
            expanded.append(output)

    result = pa.Table.from_pylist(expanded)
    metadata = {
        key: value
        for key, value in (source.schema.metadata or {}).items()
        if key != b"pandas"
    }
    metadata.update(
        {
            b"spatialcraft_transform": b"sat_test_circular_expansion",
            b"spatialcraft_source_rows": b"150",
            b"spatialcraft_circular_shifts": b"2",
        }
    )
    result = result.replace_schema_metadata(metadata)

    labels = Counter(result.column("correct_answer_label").to_pylist())
    source_counts = Counter(result.column("source_row_index").to_pylist())
    if result.num_rows != 300 or labels != {"A": 150, "B": 150}:
        raise RuntimeError(
            f"invalid circular expansion: rows={result.num_rows}, labels={labels}"
        )
    if set(source_counts.values()) != {2} or len(source_counts) != 150:
        raise RuntimeError("each source row must produce exactly two rotations")

    temporary = test_path.with_suffix(test_path.suffix + ".tmp")
    derived_path = test_path.with_name("SAT_test_circular_300.parquet")
    derived_temporary = derived_path.with_suffix(derived_path.suffix + ".tmp")
    try:
        pq.write_table(result, temporary, compression="zstd", row_group_size=64)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, test_path)
        derived_temporary.unlink(missing_ok=True)
        os.link(test_path, derived_temporary)
        os.replace(derived_temporary, derived_path)
    finally:
        temporary.unlink(missing_ok=True)
        derived_temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("/l/users/xiwei.liu/benchmark/SAT/SAT_test.parquet"),
    )
    parser.add_argument(
        "--backup-path",
        type=Path,
        default=Path("/l/users/xiwei.liu/benchmark/SAT/SAT_test_original_150.parquet"),
    )
    args = parser.parse_args()
    expand(args.test_path, args.backup_path)
    print(f"wrote 300 circular test rows to {args.test_path}")
    print(
        "preserved a download-safe derived link at "
        f"{args.test_path.with_name('SAT_test_circular_300.parquet')}"
    )
    print(f"preserved 150 official rows at {args.backup_path}")


if __name__ == "__main__":
    main()
