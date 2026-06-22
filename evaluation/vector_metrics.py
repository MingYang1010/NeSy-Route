"""Shared vector metrics for NeSy-Route Task 1 and Task 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr

LAND_CLASSES = [
    "Bareland",
    "Rangeland",
    "Developed",
    "Road",
    "Tree",
    "Water",
    "Agriculture",
    "Building",
]


def normalize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def normalize_dataset_name(dataset: str) -> str:
    return dataset[:-5] if dataset.endswith(".json") else dataset


def artifact_path(directory: str, model: str, dataset: str, prompt_version: str) -> Path:
    filename = f"{normalize_model_name(model)}_{normalize_dataset_name(dataset)}_{prompt_version}.json"
    return Path(directory) / filename


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, ensure_ascii=False)


def load_prediction_file(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_json(path)
    if isinstance(data, dict):
        meta = data.get("meta")
        if not isinstance(meta, dict):
            meta = {key: value for key, value in data.items() if key != "results"}
        results = data.get("results", [])
        return meta, results
    return {}, data


def load_dataset_records(path: str | Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("matches"), list):
            return data["matches"]
        for value in data.values():
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported dataset format: {path}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    exact_match = np.all(pred == gt, axis=1)
    elementwise_accuracy = (pred == gt).mean(axis=0)

    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))
    tn = int(np.sum((pred_flat == 0) & (gt_flat == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "exact_match": float(np.mean(exact_match)),
        "elementwise_accuracy": elementwise_accuracy.tolist(),
        "mean_elementwise_accuracy": float(elementwise_accuracy.mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "error_types": {
            "false_positive": fp,
            "false_negative": fn,
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        },
    }


def ranking_correlation(pred_cost: np.ndarray, gt_cost: np.ndarray, traverse_mask: np.ndarray) -> dict[str, Any]:
    correlations: list[dict[str, float]] = []

    for pred_row, gt_row, mask_row in zip(pred_cost, gt_cost, traverse_mask, strict=True):
        mask = mask_row == 1
        if int(mask.sum()) < 2:
            continue

        pred = pred_row[mask]
        gt = gt_row[mask]
        if len(np.unique(pred)) <= 1 or len(np.unique(gt)) <= 1:
            continue

        spearman_corr, _ = spearmanr(pred, gt)
        kendall_corr, _ = kendalltau(pred, gt)
        if not np.isnan(spearman_corr) and not np.isnan(kendall_corr):
            correlations.append({"spearman": float(spearman_corr), "kendall": float(kendall_corr)})

    if not correlations:
        return {"mean_spearman": None, "mean_kendall": None, "valid_samples": 0}

    return {
        "mean_spearman": float(np.mean([item["spearman"] for item in correlations])),
        "mean_kendall": float(np.mean([item["kendall"] for item in correlations])),
        "valid_samples": len(correlations),
    }


def cost_metrics(pred_cost: np.ndarray, gt_cost: np.ndarray, gt_traverse: np.ndarray) -> dict[str, Any]:
    exact_match = np.all(pred_cost == gt_cost, axis=1)
    elementwise_accuracy = (pred_cost == gt_cost).mean(axis=0)
    traversable_mask = gt_traverse == 1

    if traversable_mask.sum() > 0:
        conditional_accuracy = float((pred_cost[traversable_mask] == gt_cost[traversable_mask]).mean())
        conditional_mae = float(np.abs(pred_cost[traversable_mask] - gt_cost[traversable_mask]).mean())
    else:
        conditional_accuracy = None
        conditional_mae = None

    pred_tiers = [
        len(np.unique(pred_cost[index][gt_traverse[index] == 1]))
        for index in range(len(pred_cost))
        if int((gt_traverse[index] == 1).sum()) > 0
    ]
    gt_tiers = [
        len(np.unique(gt_cost[index][gt_traverse[index] == 1]))
        for index in range(len(gt_cost))
        if int((gt_traverse[index] == 1).sum()) > 0
    ]

    return {
        "exact_match": float(np.mean(exact_match)),
        "elementwise_accuracy": elementwise_accuracy.tolist(),
        "mean_elementwise_accuracy": float(elementwise_accuracy.mean()),
        "mae": float(np.abs(pred_cost - gt_cost).mean()),
        "conditional_accuracy": conditional_accuracy,
        "conditional_mae": conditional_mae,
        "ranking_correlation": ranking_correlation(pred_cost, gt_cost, gt_traverse),
        "zero_consistency": float(np.mean((gt_traverse == 0) == (pred_cost == 0))),
        "priority_tier_match": float(np.mean(np.array(pred_tiers) == np.array(gt_tiers))) if pred_tiers else None,
    }


def print_metric_table(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        elif value is not None and not isinstance(value, (dict, list)):
            print(f"{key}: {value}")


def dataset_file(dataset_dir: str, dataset: str) -> Path:
    filename = dataset if dataset.endswith(".json") else f"{dataset}.json"
    return Path(dataset_dir) / filename
