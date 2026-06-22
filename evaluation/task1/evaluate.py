"""Evaluate NeSy-Route Task 1 vector predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.vector_metrics import (
    LAND_CLASSES,
    artifact_path,
    binary_metrics,
    cost_metrics,
    dataset_file,
    format_percent,
    load_dataset_records,
    load_prediction_file,
    save_json,
)


def collect_vectors(
    results: list[dict[str, Any]],
    ground_truth: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_rows: list[dict[str, Any]] = []
    pred_traverse: list[list[int]] = []
    gt_traverse: list[list[int]] = []
    pred_cost: list[list[int]] = []
    gt_cost: list[list[int]] = []

    for row in results:
        if not row.get("success", False):
            continue

        source_id = row.get("source_id")
        if source_id not in ground_truth:
            print(f"Warning: source_id {source_id} is not present in the dataset; skipped.")
            continue

        pred_t = row.get("pred_traverse_vector", [])
        pred_c = row.get("pred_cost_vector", [])
        gt_t = ground_truth[source_id].get("traverse_vector", [])
        gt_c = ground_truth[source_id].get("cost_vector", [])

        lengths = [len(pred_t), len(pred_c), len(gt_t), len(gt_c)]
        if any(length != 8 for length in lengths):
            print(f"Warning: source_id {source_id} has invalid vector lengths {lengths}; skipped.")
            continue

        valid_rows.append(row)
        pred_traverse.append(pred_t)
        pred_cost.append(pred_c)
        gt_traverse.append(gt_t)
        gt_cost.append(gt_c)

    if not valid_rows:
        raise ValueError("No valid predictions are available for evaluation.")

    return (
        valid_rows,
        np.asarray(pred_traverse),
        np.asarray(gt_traverse),
        np.asarray(pred_cost),
        np.asarray(gt_cost),
    )


def calculate_metrics(results: list[dict[str, Any]], ground_truth: dict[int, dict[str, Any]]) -> dict[str, Any]:
    valid_rows, pred_traverse, gt_traverse, pred_cost, gt_cost = collect_vectors(results, ground_truth)
    traverse_exact = np.all(pred_traverse == gt_traverse, axis=1)
    cost_exact = np.all(pred_cost == gt_cost, axis=1)

    practical_scores: list[float] = []
    for index in np.where(traverse_exact)[0]:
        mask = gt_traverse[index] == 1
        if mask.sum() > 0:
            practical_scores.append(float((pred_cost[index][mask] == gt_cost[index][mask]).mean()))

    return {
        "traverse": binary_metrics(pred_traverse, gt_traverse),
        "cost": cost_metrics(pred_cost, gt_cost, gt_traverse),
        "both_exact_match": float(np.mean(traverse_exact & cost_exact)),
        "practical_score": float(np.mean(practical_scores)) if practical_scores else None,
        "stats": {
            "total_samples": len(results),
            "valid_samples": len(valid_rows),
            "failed_samples": len(results) - len(valid_rows),
        },
    }


def print_metrics(metrics: dict[str, Any], meta: dict[str, Any]) -> None:
    stats = metrics["stats"]
    print("\nTask 1 Evaluation")
    print("=" * 80)
    print(f"Model: {meta.get('model', 'N/A')}")
    print(f"Dataset: {meta.get('dataset', 'N/A')}")
    print(f"Prompt version: {meta.get('prompt_version', 'N/A')}")
    print(f"Total / valid / failed: {stats['total_samples']} / {stats['valid_samples']} / {stats['failed_samples']}")

    traverse = metrics["traverse"]
    print("\nTraversability")
    print(f"Exact match: {format_percent(traverse['exact_match'])}")
    print(f"Mean elementwise accuracy: {format_percent(traverse['mean_elementwise_accuracy'])}")
    print(f"Precision / Recall / F1: {format_percent(traverse['precision'])} / {format_percent(traverse['recall'])} / {format_percent(traverse['f1'])}")
    print(f"Specificity: {format_percent(traverse['specificity'])}")

    cost = metrics["cost"]
    print("\nCost ranking")
    print(f"Exact match: {format_percent(cost['exact_match'])}")
    print(f"Mean elementwise accuracy: {format_percent(cost['mean_elementwise_accuracy'])}")
    print(f"MAE: {cost['mae']:.3f}")
    print(f"Conditional accuracy: {format_percent(cost['conditional_accuracy'])}")
    print(f"Conditional MAE: {cost['conditional_mae']:.3f}" if cost["conditional_mae"] is not None else "Conditional MAE: N/A")

    correlation = cost["ranking_correlation"]
    print(
        "Ranking correlation: "
        f"Spearman={correlation['mean_spearman']}, Kendall={correlation['mean_kendall']}, "
        f"valid={correlation['valid_samples']}"
    )

    print("\nPer-class traverse accuracy")
    for name, accuracy in zip(LAND_CLASSES, traverse["elementwise_accuracy"], strict=True):
        print(f"{name}: {format_percent(accuracy)}")

    print("\nSummary")
    print(f"Traverse+cost exact match: {format_percent(metrics['both_exact_match'])}")
    print(f"Practical score: {format_percent(metrics['practical_score'])}")


def build_error_report(results: list[dict[str, Any]], ground_truth: dict[int, dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in results if row.get("success", False)]
    report: dict[str, Any] = {
        "traverse_only_errors": [],
        "cost_only_errors": [],
        "both_errors": [],
        "summary": {},
    }

    for row in valid_rows:
        source_id = row.get("source_id")
        if source_id not in ground_truth:
            continue

        pred_t = np.asarray(row.get("pred_traverse_vector", []))
        pred_c = np.asarray(row.get("pred_cost_vector", []))
        gt_t = np.asarray(ground_truth[source_id].get("traverse_vector", []))
        gt_c = np.asarray(ground_truth[source_id].get("cost_vector", []))

        traverse_wrong = not np.array_equal(pred_t, gt_t)
        cost_wrong = not np.array_equal(pred_c, gt_c)
        if not traverse_wrong and not cost_wrong:
            continue

        item = {
            "source_id": source_id,
            "question": row.get("question", ""),
            "pred_traverse": pred_t.tolist(),
            "gt_traverse": gt_t.tolist(),
            "pred_cost": pred_c.tolist(),
            "gt_cost": gt_c.tolist(),
        }
        if traverse_wrong and cost_wrong:
            report["both_errors"].append(item)
        elif traverse_wrong:
            report["traverse_only_errors"].append(item)
        else:
            report["cost_only_errors"].append(item)

    total_errors = sum(len(report[key]) for key in ["traverse_only_errors", "cost_only_errors", "both_errors"])
    total_valid = len(valid_rows)
    report["summary"] = {
        "total_samples": total_valid,
        "total_errors": total_errors,
        "traverse_only_errors": len(report["traverse_only_errors"]),
        "cost_only_errors": len(report["cost_only_errors"]),
        "both_errors": len(report["both_errors"]),
        "accuracy_rate": (total_valid - total_errors) / total_valid if total_valid else 0.0,
    }
    return report


def filter_release_samples(
    results: list[dict[str, Any]],
    ground_truth: dict[int, dict[str, Any]],
    dataset_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_id = {record["source_id"]: record for record in dataset_records}
    filtered: list[dict[str, Any]] = []

    for row in results:
        if not row.get("success", False):
            continue

        source_id = row.get("source_id")
        if source_id not in ground_truth or source_id not in records_by_id:
            continue

        pred_t = np.asarray(row.get("pred_traverse_vector", []))
        pred_c = np.asarray(row.get("pred_cost_vector", []))
        gt_t = np.asarray(ground_truth[source_id].get("traverse_vector", []))
        gt_c = np.asarray(ground_truth[source_id].get("cost_vector", []))

        traverse_correct = np.array_equal(pred_t, gt_t)
        cost_correct = np.array_equal(pred_c, gt_c)
        if traverse_correct:
            filtered.append(records_by_id[source_id])

    return sorted(filtered, key=lambda item: item["source_id"])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    result_path = artifact_path(args.result_dir, args.model, args.dataset, args.prompt_version)
    dataset_path = dataset_file(args.dataset_dir, args.dataset)

    if not result_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {result_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    meta, results = load_prediction_file(result_path)
    dataset_records = load_dataset_records(dataset_path)
    ground_truth = {record["source_id"]: record for record in dataset_records}

    meta.setdefault("model", args.model)
    meta.setdefault("dataset", args.dataset)
    meta.setdefault("prompt_version", args.prompt_version)

    metrics = calculate_metrics(results, ground_truth)
    print_metrics(metrics, meta)

    metrics_path = artifact_path(args.metrics_dir, args.model, args.dataset, args.prompt_version)
    save_json(metrics_path, {"meta": meta, "metrics": metrics})
    print(f"\nSaved metrics to {metrics_path}")

    if args.errors_dir:
        errors_path = artifact_path(args.errors_dir, args.model, args.dataset, args.prompt_version)
        save_json(errors_path, build_error_report(results, ground_truth))
        print(f"Saved error report to {errors_path}")

    if args.filtered_dir:
        filtered_path = artifact_path(args.filtered_dir, args.model, args.dataset, args.prompt_version)
        save_json(filtered_path, filter_release_samples(results, ground_truth, dataset_records))
        print(f"Saved filtered samples to {filtered_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NeSy-Route Task 1 vector predictions.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--prompt_version", type=str, default="v1")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--metrics_dir", type=str, default="metrics")
    parser.add_argument("--errors_dir", type=str, default="errors")
    parser.add_argument("--filtered_dir", type=str, default="filtered_samples")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
