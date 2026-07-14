#!/usr/bin/env python3
"""Download only the annotations needed to score NeSy-Route predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Ming1010/NeSy-Route"
SPLITS = ("easy", "medium", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("task1", "task2", "task3", "all"), default="all")
    parser.add_argument("--split", choices=SPLITS, action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("data/NeSy-Route"))
    parser.add_argument("--repo-id", default=REPO_ID)
    return parser.parse_args()


def allow_patterns(task: str, splits: list[str]) -> list[str]:
    patterns: list[str] = []
    if task in {"task1", "all"}:
        patterns.append("Task1/queries.json")
    if task in {"task2", "all"}:
        patterns.extend(f"Task2/evaluation/{split}.parquet" for split in splits)
        patterns.append("Task2/evaluation/manifest.json")
    if task in {"task3", "all"}:
        patterns.extend(f"Task3/evaluation/{split}.parquet" for split in splits)
        patterns.extend(
            (
                "Task3/evaluation/manifest.json",
                "Task3/labels/labels.parquet",
                "Task3/labels/manifest.json",
            )
        )
    return patterns


def main() -> None:
    args = parse_args()
    splits = args.split or list(SPLITS)
    patterns = allow_patterns(args.task, splits)
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.output_dir,
            allow_patterns=patterns,
        )
    except Exception as error:
        raise SystemExit(
            f"Hugging Face download failed: {error}\n"
            "Check the network/proxy connection and rerun the same command; completed files are reused."
        ) from None
    print(f"Evaluation data is ready at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
