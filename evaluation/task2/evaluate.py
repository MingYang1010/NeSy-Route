"""Evaluate Task 2 predictions against a public difficulty split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.data import load_parquet_records, load_predictions, require_file, save_json, successful
from evaluation.vector_metrics import LAND_CLASSES, binary_metrics, cost_metrics, format_percent


def prediction_sample_id(row: dict[str, Any]) -> str | None:
    value = row.get("sample_id", row.get("id"))
    return str(value) if value is not None else None


def collect_vectors(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    valid_rows: list[dict[str, Any]] = []
    vectors: dict[str, list[list[int]]] = {
        "pred_traverse": [],
        "gt_traverse": [],
        "pred_cost": [],
        "gt_cost": [],
        "pred_region": [],
        "gt_region": [],
    }

    for row in predictions:
        if not successful(row):
            continue
        sample_id = prediction_sample_id(row)
        if sample_id not in ground_truth:
            print(f"Warning: unknown sample_id {sample_id!r}; skipped.")
            continue
        gt = ground_truth[sample_id]
        current = {
            "pred_traverse": row.get("pred_traverse_vector", row.get("traverse_vector", [])),
            "gt_traverse": gt.get("traverse_vector", []),
            "pred_cost": row.get("pred_cost_vector", row.get("cost_vector", [])),
            "gt_cost": gt.get("cost_vector", []),
            "pred_region": row.get("pred_region_vector", row.get("region_vector", [])),
            "gt_region": gt.get("region_vector", []),
        }
        if any(len(vector) != 8 for vector in current.values()):
            print(f"Warning: sample_id {sample_id} has a non-8D vector; skipped.")
            continue
        valid_rows.append(row)
        for key, value in current.items():
            vectors[key].append(value)

    if not valid_rows:
        raise ValueError("No valid Task 2 predictions were found.")
    return (
        valid_rows,
        np.asarray(vectors["pred_traverse"]),
        np.asarray(vectors["gt_traverse"]),
        np.asarray(vectors["pred_cost"]),
        np.asarray(vectors["gt_cost"]),
        np.asarray(vectors["pred_region"]),
        np.asarray(vectors["gt_region"]),
    )


def calculate_metrics(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    (
        valid_rows,
        pred_traverse,
        gt_traverse,
        pred_cost,
        gt_cost,
        pred_region,
        gt_region,
    ) = collect_vectors(predictions, ground_truth)

    traverse = binary_metrics(pred_traverse, gt_traverse)
    preference = cost_metrics(pred_cost, gt_cost, gt_traverse)
    region = binary_metrics(pred_region, gt_region)
    all_exact = (
        np.all(pred_traverse == gt_traverse, axis=1)
        & np.all(pred_cost == gt_cost, axis=1)
        & np.all(pred_region == gt_region, axis=1)
    )
    return {
        "paper_metrics": {
            "RM": region["mean_elementwise_accuracy"],
            "TM": traverse["exact_match"],
            "PR": preference["ranking_correlation"]["mean_kendall"],
        },
        "all_vector_exact_match": float(np.mean(all_exact)),
        "region": region,
        "traversability": traverse,
        "preference": preference,
        "stats": {
            "submitted_predictions": len(predictions),
            "valid_predictions": len(valid_rows),
            "skipped_predictions": len(predictions) - len(valid_rows),
            "ground_truth_samples": len(ground_truth),
        },
    }


def error_report(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    errors = []
    for row in predictions:
        sample_id = prediction_sample_id(row)
        if not successful(row) or sample_id not in ground_truth:
            continue
        gt = ground_truth[sample_id]
        fields = ("traverse_vector", "cost_vector", "region_vector")
        if any(row.get(f"pred_{field}", row.get(field, [])) != gt[field] for field in fields):
            item: dict[str, Any] = {"sample_id": sample_id}
            for field in fields:
                item[f"pred_{field}"] = row.get(f"pred_{field}", row.get(field, []))
                item[f"gt_{field}"] = gt[field]
            errors.append(item)
    return errors


def print_metrics(metrics: dict[str, Any], split: str) -> None:
    paper = metrics["paper_metrics"]
    stats = metrics["stats"]
    print(f"\nTask 2: Text-Image Constraint Alignment ({split})")
    print("=" * 58)
    print(f"RM (Region Matching Rate): {format_percent(paper['RM'])}")
    print(f"TM (Traversability Matching): {format_percent(paper['TM'])}")
    print(f"PR (Kendall preference correlation): {paper['PR']}")
    print(f"All-vector exact match: {format_percent(metrics['all_vector_exact_match'])}")
    print(
        "Valid / submitted / ground truth: "
        f"{stats['valid_predictions']} / {stats['submitted_predictions']} / {stats['ground_truth_samples']}"
    )
    print("\nPer-class region accuracy")
    for name, score in zip(LAND_CLASSES, metrics["region"]["elementwise_accuracy"], strict=True):
        print(f"{name}: {format_percent(score)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("easy", "medium", "hard"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/NeSy-Route"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--errors-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.data_root / "Task2" / "evaluation" / f"{args.split}.parquet"
    hint = f"Run: python scripts/download_evaluation_data.py --task task2 --split {args.split}"
    require_file(dataset_path, hint)
    require_file(args.predictions)

    output = args.output or Path(f"outputs/task2/{args.split}_metrics.json")
    errors_output = args.errors_output or Path(f"outputs/task2/{args.split}_errors.json")
    meta, predictions = load_predictions(args.predictions)
    records = load_parquet_records(dataset_path)
    ground_truth = {record["sample_id"]: record for record in records}
    metrics = calculate_metrics(predictions, ground_truth)
    print_metrics(metrics, args.split)

    save_json(output, {"meta": meta, "split": args.split, "metrics": metrics})
    save_json(errors_output, error_report(predictions, ground_truth))
    print(f"\nMetrics: {output}")
    print(f"Errors: {errors_output}")


if __name__ == "__main__":
    main()
