"""Core metrics for NeSy-Route Task 3."""

from __future__ import annotations

import heapq
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


IMPASSABLE_COST = np.uint16(65535)


def prediction_sample_id(row: dict[str, Any]) -> str | None:
    value = row.get("sample_id", row.get("id"))
    return str(value) if value is not None else None


def prediction_waypoints(row: dict[str, Any]) -> list[list[int]]:
    raw = row.get("trajectory", row.get("pred_trajectory", []))
    if not isinstance(raw, list):
        raise ValueError("trajectory must be a list of [x, y] coordinates")
    waypoints: list[list[int]] = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Invalid waypoint: {point!r}")
        x, y = point
        if not isinstance(x, (int, np.integer)) or not isinstance(y, (int, np.integer)):
            raise ValueError(f"Waypoint coordinates must be integers: {point!r}")
        waypoints.append([int(x), int(y)])
    if not waypoints:
        raise ValueError("trajectory is empty")
    return waypoints


def build_maps(
    labels: np.ndarray,
    traverse_vector: list[int],
    cost_vector: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D semantic label mask, got {labels.shape}")
    if len(traverse_vector) != 8 or len(cost_vector) != 8:
        raise ValueError("Task 3 traverse_vector and cost_vector must both have length 8")

    traverse_map = np.zeros(labels.shape, dtype=np.uint8)
    cost_map = np.full(labels.shape, IMPASSABLE_COST, dtype=np.uint16)
    max_priority = max(cost_vector)
    for class_id in range(1, 9):
        vector_index = class_id - 1
        if traverse_vector[vector_index] != 1:
            continue
        class_mask = labels == class_id
        traverse_map[class_mask] = 1
        cost_map[class_mask] = max_priority - cost_vector[vector_index] + 1
    return traverse_map, cost_map


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    points: list[list[int]] = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx - dy
    x, y = x0, y0
    while True:
        points.append([x, y])
        if x == x1 and y == y1:
            return points
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x += sx
        if doubled < dx:
            error += dx
            y += sy


def connect_with_bresenham(waypoints: list[list[int]]) -> list[list[int]]:
    if len(waypoints) == 1:
        return waypoints.copy()
    trajectory: list[list[int]] = []
    for index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        segment = bresenham_line(*start, *end)
        trajectory.extend(segment if index == 0 else segment[1:])
    return trajectory


def _heuristic(point: tuple[int, int], goal: tuple[int, int], min_cost: float) -> float:
    return min_cost * math.hypot(point[0] - goal[0], point[1] - goal[1]) / math.sqrt(2.0)


def astar_segment(
    start: list[int],
    goal: list[int],
    cost_map: np.ndarray,
    min_cost: float | None = None,
) -> list[list[int]]:
    start_point, goal_point = tuple(start), tuple(goal)
    if start_point == goal_point:
        return [start.copy()]

    height, width = cost_map.shape
    for x, y in (start_point, goal_point):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"A* endpoint is out of bounds: {[x, y]}")
        if cost_map[y, x] >= IMPASSABLE_COST:
            raise ValueError(f"A* endpoint is non-traversable: {[x, y]}")

    if min_cost is None:
        passable = cost_map[cost_map < IMPASSABLE_COST]
        min_cost = float(passable.min())
    queue: list[tuple[float, int, tuple[int, int]]] = [(0.0, 0, start_point)]
    counter = 1
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start_point: 0.0}
    closed: set[tuple[int, int]] = set()

    while queue:
        _, _, current = heapq.heappop(queue)
        if current == goal_point:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return [[x, y] for x, y in path]
        if current in closed:
            continue
        closed.add(current)

        x, y = current
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                nx, ny = neighbor
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                terrain_cost = cost_map[ny, nx]
                if terrain_cost >= IMPASSABLE_COST:
                    continue
                tentative = g_score[current] + float(terrain_cost)
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                score = tentative + _heuristic(neighbor, goal_point, min_cost)
                heapq.heappush(queue, (score, counter, neighbor))
                counter += 1
    raise RuntimeError(f"No traversable path between {start} and {goal}")


def connect_with_astar(waypoints: list[list[int]], cost_map: np.ndarray) -> list[list[int]]:
    if len(waypoints) == 1:
        return waypoints.copy()
    passable = cost_map[cost_map < IMPASSABLE_COST]
    min_cost = float(passable.min())
    trajectory: list[list[int]] = []
    for index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        segment = astar_segment(start, end, cost_map, min_cost)
        trajectory.extend(segment if index == 0 else segment[1:])
    return trajectory


def waypoint_feasibility(
    waypoints: list[list[int]],
    traverse_map: np.ndarray,
    labels: np.ndarray,
) -> tuple[bool, list[dict[str, Any]]]:
    height, width = traverse_map.shape
    failures = []
    for index, (x, y) in enumerate(waypoints):
        if not (0 <= x < width and 0 <= y < height):
            failures.append({"index": index, "point": [x, y], "reason": "out_of_bounds"})
        elif labels[y, x] == 0:
            failures.append({"index": index, "point": [x, y], "reason": "empty"})
        elif traverse_map[y, x] == 0:
            failures.append(
                {
                    "index": index,
                    "point": [x, y],
                    "reason": "non_traversable",
                    "label": int(labels[y, x]),
                }
            )
    return not failures, failures


def path_cost(trajectory: list[list[int]], cost_map: np.ndarray) -> float:
    height, width = cost_map.shape
    total = 0.0
    for x, y in trajectory:
        if 0 <= x < width and 0 <= y < height:
            total += float(cost_map[y, x])
        else:
            total += float(IMPASSABLE_COST)
    return total


def violation_ratio(trajectory: list[list[int]], traverse_map: np.ndarray) -> float:
    if not trajectory:
        return 1.0
    height, width = traverse_map.shape
    violations = 0
    for x, y in trajectory:
        if not (0 <= x < width and 0 <= y < height) or traverse_map[y, x] == 0:
            violations += 1
    return violations / len(trajectory)


def euclidean_distance(first: list[int], second: list[int]) -> float:
    return float(math.hypot(first[0] - second[0], first[1] - second[1]))


def _directed_mean_min_distance(first: np.ndarray, second: np.ndarray) -> float:
    total = 0.0
    chunk_size = 512
    for start in range(0, len(first), chunk_size):
        chunk = first[start : start + chunk_size]
        squared = ((chunk[:, None, :] - second[None, :, :]) ** 2).sum(axis=-1)
        total += float(np.sqrt(squared.min(axis=1)).sum())
    return total / len(first)


def chamfer_distance(trajectory: list[list[int]], gt_trajectory: list[list[int]]) -> float:
    if not trajectory or not gt_trajectory:
        return float("inf")
    predicted = np.asarray(trajectory, dtype=np.float32)
    ground_truth = np.asarray(gt_trajectory, dtype=np.float32)
    return (
        _directed_mean_min_distance(predicted, ground_truth)
        + _directed_mean_min_distance(ground_truth, predicted)
    ) / 2.0


def label_cache_name(source_image_name: str) -> str:
    return f"{Path(source_image_name).stem}.png"


def materialize_label_cache(label_parquet: Path, cache_dir: Path) -> int:
    cache_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(label_parquet, columns=["source_image_name", "label"])
    written = 0
    for row in table.to_pylist():
        target = cache_dir / label_cache_name(row["source_image_name"])
        if target.is_file():
            continue
        payload = row["label"]["bytes"]
        if payload is None:
            raise ValueError(f"Missing label bytes for {row['source_image_name']}")
        temporary = target.with_suffix(".png.incomplete")
        temporary.write_bytes(payload)
        with Image.open(io.BytesIO(payload)) as image:
            if image.size != (row.get("width", image.width), row.get("height", image.height)):
                raise ValueError(f"Invalid label dimensions for {row['source_image_name']}")
        temporary.replace(target)
        written += 1
    return written


def evaluate_sample(
    prediction: dict[str, Any],
    sample: dict[str, Any],
    label_cache_dir: str,
) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    result: dict[str, Any] = {"sample_id": sample_id, "query_id": sample["query_id"]}
    try:
        waypoints = prediction_waypoints(prediction)
        label_path = Path(label_cache_dir) / label_cache_name(sample["source_image_name"])
        with Image.open(label_path) as image:
            labels = np.asarray(image)
        traverse_map, cost_map = build_maps(
            labels,
            sample["traverse_vector"],
            sample["cost_vector"],
        )
        adherent, failures = waypoint_feasibility(waypoints, traverse_map, labels)
        result.update(
            {
                "num_waypoints": len(waypoints),
                "adherent": adherent,
                "infeasible_waypoints": failures[:100],
                "start_error": euclidean_distance(waypoints[0], sample["start_point"]),
                "end_error": euclidean_distance(waypoints[-1], sample["end_point"]),
            }
        )

        if adherent:
            try:
                dense_trajectory = connect_with_astar(waypoints, cost_map)
                gt_cost = path_cost(sample["gt_trajectory"], cost_map)
                result["connection_method"] = "astar"
                result["connection_success"] = True
                result["cost_ratio"] = path_cost(dense_trajectory, cost_map) / gt_cost
            except Exception as error:
                dense_trajectory = connect_with_bresenham(waypoints)
                result["connection_method"] = "bresenham_fallback"
                result["connection_success"] = False
                result["connection_error"] = str(error)
        else:
            dense_trajectory = connect_with_bresenham(waypoints)
            result["connection_method"] = "bresenham"
            result["connection_success"] = True
            result["violation_ratio"] = violation_ratio(dense_trajectory, traverse_map)

        result["dense_path_length"] = len(dense_trajectory)
        result["gt_path_length"] = len(sample["gt_trajectory"])
        result["chamfer_distance"] = chamfer_distance(
            dense_trajectory,
            sample["gt_trajectory"],
        )
    except Exception as error:
        result["error"] = str(error)
    return result


def aggregate_metrics(results: list[dict[str, Any]], ground_truth_count: int) -> dict[str, Any]:
    if not results:
        raise ValueError("No matched Task 3 predictions were evaluated")
    successful_results = [row for row in results if "error" not in row]
    cost_ratios = [
        row["cost_ratio"]
        for row in successful_results
        if row.get("adherent") and "cost_ratio" in row
    ]
    violation_ratios = [
        row["violation_ratio"]
        for row in successful_results
        if not row.get("adherent") and "violation_ratio" in row
    ]
    chamfer_values = [
        row["chamfer_distance"]
        for row in successful_results
        if math.isfinite(row.get("chamfer_distance", float("inf")))
    ]
    return {
        "paper_metrics": {
            "AR": sum(bool(row.get("adherent")) for row in results) / len(results),
            "CR": float(np.mean(cost_ratios)) if cost_ratios else None,
            "VR": float(np.mean(violation_ratios)) if violation_ratios else None,
            "CD": float(np.mean(chamfer_values)) if chamfer_values else None,
        },
        "stats": {
            "ground_truth_samples": ground_truth_count,
            "matched_predictions": len(results),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(results) - len(successful_results),
            "adherent_with_cost_ratio": len(cost_ratios),
            "non_adherent_with_violation_ratio": len(violation_ratios),
        },
    }
