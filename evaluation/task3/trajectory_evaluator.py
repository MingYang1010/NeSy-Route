"""Evaluate Task 3 trajectories using the public HF annotations and labels."""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.data import (
    load_parquet_records,
    load_predictions,
    require_file,
    save_json,
    save_jsonl,
    successful,
)
from evaluation.task3.trajectory_core import (
    aggregate_metrics,
    evaluate_sample,
    materialize_label_cache,
    prediction_sample_id,
)


def _worker(arguments: tuple[dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    return evaluate_sample(*arguments)


def print_summary(summary: dict[str, Any], split: str) -> None:
    paper = summary["paper_metrics"]
    stats = summary["stats"]
    print(f"\nTask 3: Constrained Route Planning ({split})")
    print("=" * 51)
    print(f"AR (Adherence Rate): {paper['AR']:.6f}")
    print(f"CR (Cost Ratio): {paper['CR']}")
    print(f"VR (Violation Ratio): {paper['VR']}")
    print(f"CD (Chamfer Distance): {paper['CD']}")
    print(
        "Matched / successful / ground truth: "
        f"{stats['matched_predictions']} / {stats['successful_evaluations']} / "
        f"{stats['ground_truth_samples']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("easy", "medium", "hard"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/NeSy-Route"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task3"))
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require exactly one prediction for every sample in the selected split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")

    annotations_path = args.data_root / "Task3" / "evaluation" / f"{args.split}.parquet"
    labels_path = args.data_root / "Task3" / "labels" / "labels.parquet"
    hint = f"Run: python scripts/download_evaluation_data.py --task task3 --split {args.split}"
    require_file(annotations_path, hint)
    require_file(labels_path, hint)
    require_file(args.predictions)

    meta, predictions = load_predictions(args.predictions)
    annotations = load_parquet_records(annotations_path)
    ground_truth = {row["sample_id"]: row for row in annotations}

    prediction_map: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        if not successful(prediction):
            continue
        sample_id = prediction_sample_id(prediction)
        if sample_id is None:
            print("Warning: prediction without sample_id/id was skipped.")
            continue
        if sample_id in prediction_map:
            raise ValueError(f"Duplicate prediction for sample_id {sample_id}")
        prediction_map[sample_id] = prediction

    unknown = sorted(set(prediction_map) - set(ground_truth))
    if unknown:
        print(f"Warning: {len(unknown)} prediction IDs are outside the {args.split} split and were skipped.")
    matched_ids = [sample_id for sample_id in ground_truth if sample_id in prediction_map]
    if not matched_ids:
        raise ValueError(f"No predictions match the Task 3 {args.split} split")
    if args.strict and len(matched_ids) != len(ground_truth):
        missing = len(ground_truth) - len(matched_ids)
        raise ValueError(f"Strict evaluation requires the complete split; {missing} predictions are missing")

    label_cache = args.data_root / "Task3" / ".label_cache"
    written = materialize_label_cache(labels_path, label_cache)
    if written:
        print(f"Materialized {written} semantic masks at {label_cache}")

    work = [
        (prediction_map[sample_id], ground_truth[sample_id], str(label_cache))
        for sample_id in matched_ids
    ]
    if args.num_workers == 1:
        results = [_worker(item) for item in tqdm(work, desc="Task 3 evaluation", unit="sample")]
    else:
        with Pool(processes=args.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_worker, work),
                    total=len(work),
                    desc="Task 3 evaluation",
                    unit="sample",
                )
            )

    summary = aggregate_metrics(results, len(ground_truth))
    summary["stats"].update(
        {
            "submitted_predictions": len(predictions),
            "unknown_prediction_ids": len(unknown),
            "split_coverage": len(matched_ids) / len(ground_truth),
        }
    )
    print_summary(summary, args.split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.output_dir / f"{args.split}_per_sample.jsonl"
    summary_path = args.output_dir / f"{args.split}_metrics.json"
    save_jsonl(per_sample_path, results)
    save_json(summary_path, {"meta": meta, "split": args.split, "metrics": summary})
    print(f"\nMetrics: {summary_path}")
    print(f"Per-sample results: {per_sample_path}")


if __name__ == "__main__":
    main()
