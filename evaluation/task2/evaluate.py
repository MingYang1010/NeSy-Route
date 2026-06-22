"""Evaluate NeSy-Route Task 2 vector and region predictions."""

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
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_rows: list[dict[str, Any]] = []
    pred_traverse: list[list[int]] = []
    gt_traverse: list[list[int]] = []
    pred_cost: list[list[int]] = []
    gt_cost: list[list[int]] = []
    pred_region: list[list[int]] = []
    gt_region: list[list[int]] = []

    for row in results:
        if not row.get("success", False):
            continue

        sample_id = row.get("id")
        if sample_id not in ground_truth:
            print(f"Warning: id {sample_id} is not present in the dataset; skipped.")
            continue

        vectors = {
            "pred_traverse": row.get("pred_traverse_vector", []),
            "pred_cost": row.get("pred_cost_vector", []),
            "pred_region": row.get("pred_region_vector", []),
            "gt_traverse": ground_truth[sample_id].get("traverse_vector", []),
            "gt_cost": ground_truth[sample_id].get("cost_vector", []),
            "gt_region": ground_truth[sample_id].get("region_vector", []),
        }
        if any(len(value) != 8 for value in vectors.values()):
            lengths = {key: len(value) for key, value in vectors.items()}
            print(f"Warning: id {sample_id} has invalid vector lengths {lengths}; skipped.")
            continue

        valid_rows.append(row)
        pred_traverse.append(vectors["pred_traverse"])
        pred_cost.append(vectors["pred_cost"])
        pred_region.append(vectors["pred_region"])
        gt_traverse.append(vectors["gt_traverse"])
        gt_cost.append(vectors["gt_cost"])
        gt_region.append(vectors["gt_region"])

    if not valid_rows:
        raise ValueError("No valid predictions are available for evaluation.")

    return (
        valid_rows,
        np.asarray(pred_traverse),
        np.asarray(gt_traverse),
        np.asarray(pred_cost),
        np.asarray(gt_cost),
        np.asarray(pred_region),
        np.asarray(gt_region),
    )


def region_metrics(pred_region: np.ndarray, gt_region: np.ndarray) -> dict[str, Any]:
    metrics = binary_metrics(pred_region, gt_region)
    target_mask = gt_region == 1

    if target_mask.sum() == 0:
        metrics.update(
            {
                "target_region_accuracy": None,
                "target_region_accuracy_per_sample": None,
                "target_region_accuracy_per_position": [None] * gt_region.shape[1],
                "target_region_count_per_position": [0] * gt_region.shape[1],
            }
        )
        return metrics

    per_sample = []
    for index in range(len(pred_region)):
        sample_mask = gt_region[index] == 1
        if sample_mask.sum() > 0:
            per_sample.append(float((pred_region[index][sample_mask] == gt_region[index][sample_mask]).mean()))

    per_position: list[float | None] = []
    count_per_position: list[int] = []
    for position in range(gt_region.shape[1]):
        position_mask = gt_region[:, position] == 1
        count = int(position_mask.sum())
        count_per_position.append(count)
        if count:
            per_position.append(float((pred_region[:, position][position_mask] == gt_region[:, position][position_mask]).mean()))
        else:
            per_position.append(None)

    metrics.update(
        {
            "target_region_accuracy": float((pred_region[target_mask] == gt_region[target_mask]).mean()),
            "target_region_accuracy_per_sample": {
                "mean": float(np.mean(per_sample)),
                "std": float(np.std(per_sample)),
                "min": float(np.min(per_sample)),
                "max": float(np.max(per_sample)),
            },
            "target_region_accuracy_per_position": per_position,
            "target_region_count_per_position": count_per_position,
        }
    )
    return metrics


def calculate_metrics(results: list[dict[str, Any]], ground_truth: dict[int, dict[str, Any]]) -> dict[str, Any]:
    (
        valid_rows,
        pred_traverse,
        gt_traverse,
        pred_cost,
        gt_cost,
        pred_region,
        gt_region,
    ) = collect_vectors(results, ground_truth)

    traverse_exact = np.all(pred_traverse == gt_traverse, axis=1)
    cost_exact = np.all(pred_cost == gt_cost, axis=1)
    region_exact = np.all(pred_region == gt_region, axis=1)

    practical_scores: list[float] = []
    for index in np.where(traverse_exact)[0]:
        mask = gt_traverse[index] == 1
        if mask.sum() > 0:
            practical_scores.append(float((pred_cost[index][mask] == gt_cost[index][mask]).mean()))

    return {
        "traverse": binary_metrics(pred_traverse, gt_traverse),
        "cost": cost_metrics(pred_cost, gt_cost, gt_traverse),
        "region": region_metrics(pred_region, gt_region),
        "both_exact_match": float(np.mean(traverse_exact & cost_exact)),
        "all_exact_match": float(np.mean(traverse_exact & cost_exact & region_exact)),
        "practical_score": float(np.mean(practical_scores)) if practical_scores else None,
        "stats": {
            "total_samples": len(results),
            "valid_samples": len(valid_rows),
            "failed_samples": len(results) - len(valid_rows),
        },
    }


def print_metrics(metrics: dict[str, Any], meta: dict[str, Any]) -> None:
    stats = metrics["stats"]
    print("\nTask 2 Evaluation")
    print("=" * 80)
    print(f"Model: {meta.get('model', 'N/A')}")
    print(f"Dataset: {meta.get('dataset', 'N/A')}")
    print(f"Prompt version: {meta.get('prompt_version', 'N/A')}")
    print(f"Total / valid / failed: {stats['total_samples']} / {stats['valid_samples']} / {stats['failed_samples']}")

    for section_name, key in [("Traversability", "traverse"), ("Cost ranking", "cost"), ("Region recognition", "region")]:
        section = metrics[key]
        print(f"\n{section_name}")
        print(f"Exact match: {format_percent(section['exact_match'])}")
        print(f"Mean elementwise accuracy: {format_percent(section['mean_elementwise_accuracy'])}")
        if key != "cost":
            print(f"Precision / Recall / F1: {format_percent(section['precision'])} / {format_percent(section['recall'])} / {format_percent(section['f1'])}")
        else:
            print(f"MAE: {section['mae']:.3f}")
            print(f"Conditional accuracy: {format_percent(section['conditional_accuracy'])}")
            print(f"Conditional MAE: {section['conditional_mae']:.3f}" if section["conditional_mae"] is not None else "Conditional MAE: N/A")

    region = metrics["region"]
    print(f"\nTarget-region accuracy: {format_percent(region['target_region_accuracy'])}")
    print("\nPer-class region accuracy")
    for name, accuracy in zip(LAND_CLASSES, region["elementwise_accuracy"], strict=True):
        print(f"{name}: {format_percent(accuracy)}")

    print("\nSummary")
    print(f"Traverse+cost exact match: {format_percent(metrics['both_exact_match'])}")
    print(f"All-vector exact match: {format_percent(metrics['all_exact_match'])}")
    print(f"Practical score: {format_percent(metrics['practical_score'])}")


def build_error_report(results: list[dict[str, Any]], ground_truth: dict[int, dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in results if row.get("success", False)]
    report: dict[str, Any] = {
        "traverse_only_errors": [],
        "cost_only_errors": [],
        "region_only_errors": [],
        "multiple_errors": [],
        "summary": {},
    }

    for row in valid_rows:
        sample_id = row.get("id")
        if sample_id not in ground_truth:
            continue

        pred_t = np.asarray(row.get("pred_traverse_vector", []))
        pred_c = np.asarray(row.get("pred_cost_vector", []))
        pred_r = np.asarray(row.get("pred_region_vector", []))
        gt_t = np.asarray(ground_truth[sample_id].get("traverse_vector", []))
        gt_c = np.asarray(ground_truth[sample_id].get("cost_vector", []))
        gt_r = np.asarray(ground_truth[sample_id].get("region_vector", []))

        wrong = {
            "traverse": not np.array_equal(pred_t, gt_t),
            "cost": not np.array_equal(pred_c, gt_c),
            "region": not np.array_equal(pred_r, gt_r),
        }
        if not any(wrong.values()):
            continue

        item = {
            "id": sample_id,
            "question": row.get("question", ""),
            "pred_traverse": pred_t.tolist(),
            "gt_traverse": gt_t.tolist(),
            "pred_cost": pred_c.tolist(),
            "gt_cost": gt_c.tolist(),
            "pred_region": pred_r.tolist(),
            "gt_region": gt_r.tolist(),
        }
        if sum(wrong.values()) > 1:
            report["multiple_errors"].append(item)
        elif wrong["traverse"]:
            report["traverse_only_errors"].append(item)
        elif wrong["cost"]:
            report["cost_only_errors"].append(item)
        else:
            report["region_only_errors"].append(item)

    total_errors = sum(
        len(report[key])
        for key in ["traverse_only_errors", "cost_only_errors", "region_only_errors", "multiple_errors"]
    )
    total_valid = len(valid_rows)
    report["summary"] = {
        "total_samples": total_valid,
        "total_errors": total_errors,
        "traverse_only_errors": len(report["traverse_only_errors"]),
        "cost_only_errors": len(report["cost_only_errors"]),
        "region_only_errors": len(report["region_only_errors"]),
        "multiple_errors": len(report["multiple_errors"]),
        "accuracy_rate": (total_valid - total_errors) / total_valid if total_valid else 0.0,
    }
    return report


def filter_release_samples(
    results: list[dict[str, Any]],
    ground_truth: dict[int, dict[str, Any]],
    dataset_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_id = {record["id"]: record for record in dataset_records}
    filtered: list[dict[str, Any]] = []

    for row in results:
        if not row.get("success", False):
            continue

        sample_id = row.get("id")
        if sample_id not in ground_truth or sample_id not in records_by_id:
            continue

        pred_t = np.asarray(row.get("pred_traverse_vector", []))
        pred_c = np.asarray(row.get("pred_cost_vector", []))
        pred_r = np.asarray(row.get("pred_region_vector", []))
        gt_t = np.asarray(ground_truth[sample_id].get("traverse_vector", []))
        gt_c = np.asarray(ground_truth[sample_id].get("cost_vector", []))
        gt_r = np.asarray(ground_truth[sample_id].get("region_vector", []))

        traverse_correct = np.array_equal(pred_t, gt_t)
        cost_correct = np.array_equal(pred_c, gt_c)
        region_correct = np.array_equal(pred_r, gt_r)
        if traverse_correct and region_correct:
            filtered.append(records_by_id[sample_id])

    return sorted(filtered, key=lambda item: item["id"])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    result_path = artifact_path(args.result_dir, args.model, args.dataset, args.prompt_version)
    dataset_path = dataset_file(args.dataset_dir, args.dataset)

    if not result_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {result_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    meta, results = load_prediction_file(result_path)
    dataset_records = load_dataset_records(dataset_path)
    ground_truth = {record["id"]: record for record in dataset_records}

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
    parser = argparse.ArgumentParser(description="Evaluate NeSy-Route Task 2 vector and region predictions.")
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
