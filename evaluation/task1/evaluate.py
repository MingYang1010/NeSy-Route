"""Evaluate Task 1 predictions against the public Hugging Face annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.data import load_json_records, load_predictions, require_file, save_json, successful
from evaluation.vector_metrics import LAND_CLASSES, binary_metrics, cost_metrics, format_percent


def prediction_query_id(row: dict[str, Any]) -> str | None:
    value = row.get("query_id", row.get("source_id"))
    if isinstance(value, int):
        return f"Q{value:05d}"
    return str(value) if value is not None else None


def collect_vectors(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_rows: list[dict[str, Any]] = []
    pred_traverse: list[list[int]] = []
    gt_traverse: list[list[int]] = []
    pred_cost: list[list[int]] = []
    gt_cost: list[list[int]] = []

    for row in predictions:
        if not successful(row):
            continue
        query_id = prediction_query_id(row)
        if query_id not in ground_truth:
            print(f"Warning: unknown query_id {query_id!r}; skipped.")
            continue

        vectors = (
            row.get("pred_traverse_vector", row.get("traverse_vector", [])),
            ground_truth[query_id].get("traverse_vector", []),
            row.get("pred_cost_vector", row.get("cost_vector", [])),
            ground_truth[query_id].get("cost_vector", []),
        )
        if any(len(vector) != 8 for vector in vectors):
            print(f"Warning: query_id {query_id} has a non-8D vector; skipped.")
            continue

        valid_rows.append(row)
        pred_traverse.append(vectors[0])
        gt_traverse.append(vectors[1])
        pred_cost.append(vectors[2])
        gt_cost.append(vectors[3])

    if not valid_rows:
        raise ValueError("No valid Task 1 predictions were found.")
    return (
        valid_rows,
        np.asarray(pred_traverse),
        np.asarray(gt_traverse),
        np.asarray(pred_cost),
        np.asarray(gt_cost),
    )


def calculate_metrics(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    valid_rows, pred_traverse, gt_traverse, pred_cost, gt_cost = collect_vectors(
        predictions, ground_truth
    )
    traverse = binary_metrics(pred_traverse, gt_traverse)
    cost = cost_metrics(pred_cost, gt_cost, gt_traverse)
    fully_matching = np.all(pred_traverse == gt_traverse, axis=1) & np.all(
        pred_cost == gt_cost, axis=1
    )
    return {
        "paper_metrics": {
            "TM": traverse["exact_match"],
            "PR": cost["ranking_correlation"]["mean_kendall"],
            "FM": float(np.mean(fully_matching)),
        },
        "traversability": traverse,
        "preference": cost,
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
        query_id = prediction_query_id(row)
        if not successful(row) or query_id not in ground_truth:
            continue
        gt = ground_truth[query_id]
        pred_t = row.get("pred_traverse_vector", row.get("traverse_vector", []))
        pred_c = row.get("pred_cost_vector", row.get("cost_vector", []))
        if pred_t != gt["traverse_vector"] or pred_c != gt["cost_vector"]:
            errors.append(
                {
                    "query_id": query_id,
                    "pred_traverse_vector": pred_t,
                    "gt_traverse_vector": gt["traverse_vector"],
                    "pred_cost_vector": pred_c,
                    "gt_cost_vector": gt["cost_vector"],
                }
            )
    return errors


def print_metrics(metrics: dict[str, Any]) -> None:
    paper = metrics["paper_metrics"]
    stats = metrics["stats"]
    print("\nTask 1: Textual Constraint Understanding")
    print("=" * 52)
    print(f"TM (Traversability Matching): {format_percent(paper['TM'])}")
    print(f"PR (Kendall preference correlation): {paper['PR']}")
    print(f"FM (Fully Matching Accuracy): {format_percent(paper['FM'])}")
    print(
        "Valid / submitted / ground truth: "
        f"{stats['valid_predictions']} / {stats['submitted_predictions']} / {stats['ground_truth_samples']}"
    )
    print("\nPer-class traversability accuracy")
    for name, score in zip(
        LAND_CLASSES,
        metrics["traversability"]["elementwise_accuracy"],
        strict=True,
    ):
        print(f"{name}: {format_percent(score)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/NeSy-Route/Task1/queries.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/task1/metrics.json"))
    parser.add_argument("--errors-output", type=Path, default=Path("outputs/task1/errors.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hint = "Run: python scripts/download_evaluation_data.py --task task1"
    dataset_path = require_file(args.dataset, hint)
    prediction_path = require_file(args.predictions)

    meta, predictions = load_predictions(prediction_path)
    records = load_json_records(dataset_path)
    ground_truth = {record["query_id"]: record for record in records}
    metrics = calculate_metrics(predictions, ground_truth)
    print_metrics(metrics)

    save_json(args.output, {"meta": meta, "metrics": metrics})
    save_json(args.errors_output, error_report(predictions, ground_truth))
    print(f"\nMetrics: {args.output}")
    print(f"Errors: {args.errors_output}")


if __name__ == "__main__":
    main()
