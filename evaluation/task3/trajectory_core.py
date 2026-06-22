"""Trajectory evaluation utilities for NeSy-Route Task 3."""


import json
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from skimage.graph import route_through_array
from tqdm import tqdm



CLASS_NAMES = {
    0: "Empty",
    1: "Bareland",
    2: "Rangeland",
    3: "Developed space",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Agriculture land",
    8: "Building",
}



def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[List[int]]:
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        points.append([x, y])
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return points


def connect_with_bresenham(waypoints: List[List[int]]) -> List[List[int]]:
    if len(waypoints) < 2:
        return waypoints

    full_trajectory = []

    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        segment = bresenham_line(x0, y0, x1, y1)


        if i > 0 and len(full_trajectory) > 0:
            segment = segment[1:]

        full_trajectory.extend(segment)

    return full_trajectory


def connect_with_astar(waypoints: List[List[int]],
                       cost_map: np.ndarray,
                       _traverse_map: np.ndarray) -> List[List[int]]:
    full_trajectory = []

    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        goal = waypoints[i + 1]


        try:
            indices, _ = route_through_array(
                cost_map,
                start=(start[1], start[0]),
                end=(goal[1], goal[0]),
                fully_connected=True,
                geometric=True
            )


            segment = [[col, row] for row, col in indices]


            if i > 0 and len(full_trajectory) > 0:
                segment = segment[1:]

            full_trajectory.extend(segment)

        except Exception as e:
            raise RuntimeError(f"A* failed between {start} and {goal}: {str(e)}")

    return full_trajectory




def check_all_waypoints_feasible(waypoints: List[List[int]],
                                 traverse_map: np.ndarray,
                                 labels: np.ndarray) -> Dict:
    height, width = traverse_map.shape
    infeasible = []

    for i, (x, y) in enumerate(waypoints):

        if not (0 <= x < width and 0 <= y < height):
            infeasible.append({
                'index': i,
                'point': [int(x), int(y)],
                'reason': 'out_of_bounds'
            })
            continue


        label = labels[y, x]
        if label == 0:
            infeasible.append({
                'index': i,
                'point': [int(x), int(y)],
                'reason': 'empty_region',
                'label': int(label)
            })
            continue


        if traverse_map[y, x] == 0:
            infeasible.append({
                'index': i,
                'point': [int(x), int(y)],
                'reason': 'non_traversable',
                'label': int(label)
            })
            continue

    return {
        'all_feasible': len(infeasible) == 0,
        'infeasible_points': infeasible
    }




def compute_path_cost(trajectory: List[List[int]], cost_map: np.ndarray) -> float:
    total_cost = 0.0
    height, width = cost_map.shape

    for x, y in trajectory:
        if 0 <= x < width and 0 <= y < height:
            cost = cost_map[y, x]

            total_cost += float(cost)
        else:
            total_cost += 65535.0

    return total_cost


def compute_euclidean_length(trajectory: List[List[int]]) -> float:
    if len(trajectory) < 2:
        return 0.0

    length = 0.0
    for i in range(len(trajectory) - 1):
        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i + 1]
        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return length


def compute_straight_distance(trajectory: List[List[int]]) -> float:
    if len(trajectory) < 2:
        return 0.0

    x1, y1 = trajectory[0]
    x2, y2 = trajectory[-1]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def euclidean_distance(p1: Optional[List[int]], p2: Optional[List[int]]) -> Optional[float]:
    if p1 is None or p2 is None:
        return None
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)




def evaluate_optimality(trajectory: List[List[int]],
                       gt_path: np.ndarray,
                       cost_map: np.ndarray) -> Dict:

    pred_cost = compute_path_cost(trajectory, cost_map)
    gt_cost = compute_path_cost(gt_path.tolist(), cost_map)


    pred_length = len(trajectory)
    gt_length = len(gt_path)


    pred_euclidean = compute_euclidean_length(trajectory)
    gt_euclidean = compute_euclidean_length(gt_path.tolist())


    straight_dist = compute_straight_distance(trajectory)

    return {
        'total_cost': float(pred_cost),
        'gt_cost': float(gt_cost),
        'cost_ratio': float(pred_cost / gt_cost if gt_cost > 0 else float('inf')),
        'path_length': int(pred_length),
        'gt_length': int(gt_length),
        'length_ratio': float(pred_length / gt_length if gt_length > 0 else float('inf')),
        'euclidean_length': float(pred_euclidean),
        'gt_euclidean_length': float(gt_euclidean),
        'straight_distance': float(straight_dist),
        'euclidean_efficiency': float(straight_dist / pred_euclidean if pred_euclidean > 0 else 0),
    }


def evaluate_constraint_violations(trajectory: List[List[int]],
                                   traverse_map: np.ndarray,
                                   labels: np.ndarray,
                                   cost_map: np.ndarray) -> Dict:
    height, width = traverse_map.shape

    violations = []
    empty_violations = 0
    non_traversable = 0
    out_of_bounds = 0

    for i, (x, y) in enumerate(trajectory):

        if not (0 <= x < width and 0 <= y < height):
            violations.append({'index': i, 'point': [int(x), int(y)], 'type': 'out_of_bounds'})
            out_of_bounds += 1
            continue


        label = labels[y, x]
        if label == 0:
            violations.append({'index': i, 'point': [int(x), int(y)], 'type': 'empty', 'label': int(label)})
            empty_violations += 1
            continue


        if traverse_map[y, x] == 0:
            violations.append({'index': i, 'point': [int(x), int(y)], 'type': 'non_traversable', 'label': int(label)})
            non_traversable += 1

    return {
        'violation_count': len(violations),
        'violation_ratio': float(len(violations) / len(trajectory) if len(trajectory) > 0 else 0),
        'empty_violation_count': empty_violations,
        'non_traversable_count': non_traversable,
        'out_of_bounds_count': out_of_bounds,
        'violation_details': violations[:100],
    }


def evaluate_common_metrics(trajectory: List[List[int]],
                           gt_path: np.ndarray,
                           waypoints: List[List[int]],
                           dataset_sample: Dict) -> Dict:
    start_gt = dataset_sample['start_end_pairs'][0]['start_point']
    end_gt = dataset_sample['start_end_pairs'][0]['end_point']

    start_pred = waypoints[0] if len(waypoints) > 0 else None
    end_pred = waypoints[-1] if len(waypoints) > 0 else None

    return {
        'start_error': euclidean_distance(start_pred, start_gt),
        'end_error': euclidean_distance(end_pred, end_gt),
    }






def evaluate_chamfer_distance(trajectory: List[List[int]],
                              gt_path: np.ndarray) -> Dict:
    if len(trajectory) < 2 or len(gt_path) < 2:
        return {
            'chamfer_distance': float('inf'),
            'mean_min_distance_pred_to_gt': float('inf'),
            'mean_min_distance_gt_to_pred': float('inf'),
        }

    pred = np.array(trajectory, dtype=np.float32)
    gt = np.array(gt_path, dtype=np.float32)


    def min_distances(A, B):
        dist_matrix = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1))
        return np.min(dist_matrix, axis=1)

    min_dists_p2g = min_distances(pred, gt)
    min_dists_g2p = min_distances(gt, pred)

    chamfer = (np.mean(min_dists_p2g) + np.mean(min_dists_g2p)) / 2

    return {
        'chamfer_distance': float(chamfer),
        'mean_min_distance_pred_to_gt': float(np.mean(min_dists_p2g)),
        'mean_min_distance_gt_to_pred': float(np.mean(min_dists_g2p)),
    }

def evaluate_single_trajectory(model_output: Dict,
                               dataset_sample: Dict,
                               cost_map: np.ndarray,
                               traverse_map: np.ndarray,
                               gt_path: np.ndarray,
                               labels: np.ndarray) -> Dict:
    waypoints = model_output.get('trajectory', [])
    sample_id = model_output['id']

    result = {
        'id': sample_id,
        'num_waypoints': len(waypoints),
    }


    if len(waypoints) == 0:
        result['error'] = 'empty_trajectory'
        return result


    feasibility_check = check_all_waypoints_feasible(waypoints, traverse_map, labels)

    result['all_waypoints_feasible'] = feasibility_check['all_feasible']
    result['infeasible_waypoints'] = feasibility_check['infeasible_points']
    result['infeasible_count'] = len(feasibility_check['infeasible_points'])


    if result['all_waypoints_feasible']:

        result['connection_method'] = 'astar'

        try:
            full_trajectory = connect_with_astar(waypoints, cost_map, traverse_map)
            result['connection_success'] = True


            optimality_metrics = evaluate_optimality(full_trajectory, gt_path, cost_map)
            result.update(optimality_metrics)

        except Exception as e:
            result['connection_success'] = False
            result['connection_error'] = str(e)

            full_trajectory = connect_with_bresenham(waypoints)
            result['fallback_to_bresenham'] = True


            violation_metrics = evaluate_constraint_violations(
                full_trajectory, traverse_map, labels, cost_map
            )
            result.update(violation_metrics)

    else:

        result['connection_method'] = 'bresenham'
        full_trajectory = connect_with_bresenham(waypoints)
        result['connection_success'] = True


        violation_metrics = evaluate_constraint_violations(
            full_trajectory, traverse_map, labels, cost_map
        )
        result.update(violation_metrics)


    common_metrics = evaluate_common_metrics(
        full_trajectory, gt_path, waypoints, dataset_sample
    )
    result.update(common_metrics)



    chamfer_metrics = evaluate_chamfer_distance(full_trajectory, gt_path)
    result.update(chamfer_metrics)

    return result




def process_single_sample(args):
    model_output, dataset_sample, dataset_root, dataset_name, labels_root = args
    sample_id = model_output['id']

    try:

        image_name = dataset_sample['image_name'].replace('.tif', '')


        cost_map_path = os.path.join(dataset_root, dataset_name, 'cost_map_npy', f'{sample_id}.npy')
        cost_map = np.load(cost_map_path)


        traverse_map_path = os.path.join(dataset_root, dataset_name, 'traverse_map_npy', f'{sample_id}.npy')
        traverse_map = np.load(traverse_map_path)


        gt_path_path = os.path.join(dataset_root, dataset_name, 'gt', f'{sample_id}.npy')
        gt_path = np.load(gt_path_path)


        labels_path = os.path.join(labels_root, f'{image_name}.tif')
        labels = np.array(Image.open(labels_path))


        result = evaluate_single_trajectory(
            model_output, dataset_sample, cost_map, traverse_map, gt_path, labels
        )
        return result

    except Exception as e:
        return {
            'id': sample_id,
            'error': f'processing_failed: {str(e)}'
        }




def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):

    dir_path = os.path.dirname(file_path)


    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def batch_evaluate(dataset_path: str,
                  model_output_path: str,
                  dataset_root: str,
                  labels_root: str,
                  output_dir: str,
                  subset_jsonl: Optional[str] = None,
                  num_workers: int = None):
    if num_workers is None:
        num_workers = cpu_count()


    model_output_basename = os.path.basename(model_output_path).replace('.jsonl', '')
    if subset_jsonl and os.path.exists(subset_jsonl):
        subset_name = os.path.basename(subset_jsonl).replace('.jsonl', '')
        output_filename = f"{model_output_basename}_{subset_name}_eval.jsonl"
    else:
        output_filename = f"{model_output_basename}_eval.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Running parallel evaluation with {num_workers} workers.")
    print(f"Loading dataset: {dataset_path}")
    dataset = load_jsonl(dataset_path)

    print(f"Loading model outputs: {model_output_path}")
    model_outputs = load_jsonl(model_output_path)


    dataset_dict = {sample['id']: sample for sample in dataset}


    desired_ids = None
    if subset_jsonl:
        print(f"Using subset file: {subset_jsonl}")
        subset_data = load_jsonl(subset_jsonl)
        desired_ids = {item['id'] for item in subset_data if 'id' in item}
        print(f"Subset sample IDs: {len(desired_ids)}")


    dataset_name = os.path.basename(dataset_path).replace('.jsonl', '')

    print(f"Dataset name: {dataset_name}")
    print(f"Output file: {output_path}")


    process_args = []
    for model_output in model_outputs:
        sample_id = model_output['id']

        if sample_id not in dataset_dict:
            continue


        if desired_ids is not None and sample_id not in desired_ids:
            continue

        dataset_sample = dataset_dict[sample_id]
        process_args.append((
            model_output,
            dataset_sample,
            dataset_root,
            dataset_name,
            labels_root
        ))

    print(f"Evaluating {len(process_args)} samples...")


    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_sample, process_args),
            total=len(process_args),
            desc="Evaluation",
            unit="sample"
        ))

    save_jsonl(results, output_path)
    print(f"\nEvaluation finished. Results saved to: {output_path}")


    errors = [r for r in results if 'error' in r]
    if errors:
        print(f"\nFound {len(errors)} failed samples.")

    print_statistics(results)
    return results


def print_statistics(results: List[Dict]):
    total = len(results)


    errors = [r for r in results if 'error' in r]
    valid = [r for r in results if 'error' not in r]


    astar_used = [r for r in valid if r.get('connection_method') == 'astar']
    bresenham_used = [r for r in valid if r.get('connection_method') == 'bresenham']


    all_feasible = [r for r in valid if r.get('all_waypoints_feasible')]

    print("\n" + "="*60)
    print("Evaluation statistics")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Failed samples: {len(errors)}")
    print(f"Valid samples: {len(valid)}")

    if not valid:
        print("="*60)
        return

    print("\nConnection methods:")
    print(f"  - A*: {len(astar_used)} ({len(astar_used)/len(valid)*100:.1f}%)")
    print(f"  - Bresenham: {len(bresenham_used)} ({len(bresenham_used)/len(valid)*100:.1f}%)")
    print("\nWaypoint feasibility:")
    print(f"  - All feasible: {len(all_feasible)} ({len(all_feasible)/len(valid)*100:.1f}%)")
    print(f"  - Any violation: {len(valid) - len(all_feasible)} ({(len(valid)-len(all_feasible))/len(valid)*100:.1f}%)")


    if astar_used:
        cost_ratios = [r['cost_ratio'] for r in astar_used if r.get('connection_success') and 'cost_ratio' in r]
        if cost_ratios:
            print("\nOptimality metrics for A* samples:")
            print(f"  - Mean cost ratio (pred/gt): {np.mean(cost_ratios):.3f}")
            print(f"  - Median cost ratio: {np.median(cost_ratios):.3f}")
            print(f"  - Min cost ratio: {np.min(cost_ratios):.3f}")
            print(f"  - Max cost ratio: {np.max(cost_ratios):.3f}")


    if bresenham_used:
        violation_ratios = [r['violation_ratio'] for r in bresenham_used if 'violation_ratio' in r]
        if violation_ratios:
            print("\nConstraint-violation metrics for Bresenham samples:")
            print(f"  - Mean violation ratio: {np.mean(violation_ratios):.3f}")
            print(f"  - Median violation ratio: {np.median(violation_ratios):.3f}")

    print("="*60)

    if valid:
        chamfer_vals = [r.get('chamfer_distance', float('inf')) for r in valid]
        valid_chamfer = [v for v in chamfer_vals if np.isfinite(v)]

        if valid_chamfer:
            print("\nTrajectory similarity metrics (Chamfer distance):")
            print(f"  - Mean Chamfer distance: {np.mean(valid_chamfer):.2f} pixels")
            print(f"  - Median Chamfer distance: {np.median(valid_chamfer):.2f} pixels")
            print(f"  - Min Chamfer distance: {np.min(valid_chamfer):.2f} pixels")

def check_data_paths(dataset_path: str,
                     model_output_path: str,
                     dataset_root: str,
                     labels_root: str):
    print("="*60)
    print("Path check")
    print("="*60)


    print(f"\n1. Dataset file: {dataset_path}")
    if os.path.exists(dataset_path):
        print("   OK")
        dataset = load_jsonl(dataset_path)
        print(f"   Samples: {len(dataset)}")
        if dataset:
            sample = dataset[0]
            print(f"   First sample ID: {sample['id']}")
            print(f"   image_name: {sample['image_name']}")
    else:
        print("   MISSING")
        return False


    print(f"\n2. Model output file: {model_output_path}")
    if os.path.exists(model_output_path):
        print("   OK")
        outputs = load_jsonl(model_output_path)
        print(f"   Outputs: {len(outputs)}")
        if outputs:
            print(f"   First output ID: {outputs[0]['id']}")
    else:
        print("   MISSING")
        return False


    dataset_name = os.path.basename(dataset_path).replace('.jsonl', '')
    print(f"\n3. Dataset root: {dataset_root}")
    print(f"   Dataset name: {dataset_name}")

    required_folders = ['cost_map_npy', 'traverse_map_npy', 'gt', 'samples']
    dataset_folder = os.path.join(dataset_root, dataset_name)

    print(f"   Dataset folder: {dataset_folder}")
    if not os.path.exists(dataset_folder):
        print("   MISSING dataset folder")
        return False

    for folder in required_folders:
        folder_path = os.path.join(dataset_folder, folder)
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            print(f"   OK {folder}: {len(files)} files")
        else:
            print(f"   MISSING {folder}")


    print(f"\n4. Label mask root: {labels_root}")
    if os.path.exists(labels_root):
        label_files = [f for f in os.listdir(labels_root) if f.endswith('.tif')]
        print(f"   OK, {len(label_files)} .tif files")
        if label_files:
            print(f"   Example: {label_files[0]}")
    else:
        print("   MISSING")
        return False


    if dataset and outputs:
        sample_id = outputs[0]['id']
        if sample_id in {s['id'] for s in dataset}:
            sample = [s for s in dataset if s['id'] == sample_id][0]
            image_name = sample['image_name'].replace('.tif', '')

            print(f"\n5. Files for sample {sample_id}:")

            files_to_check = {
                'cost_map': os.path.join(dataset_folder, 'cost_map_npy', f'{sample_id}.npy'),
                'traverse_map': os.path.join(dataset_folder, 'traverse_map_npy', f'{sample_id}.npy'),
                'gt': os.path.join(dataset_folder, 'gt', f'{sample_id}.npy'),
                'sample': os.path.join(dataset_folder, 'samples', f'{sample_id}.tif'),
                'label': os.path.join(labels_root, f'{image_name}.tif'),
            }

            for name, path in files_to_check.items():
                if os.path.exists(path):
                    print(f"   OK {name}: {path}")
                else:
                    print(f"   MISSING {name}: {path}")

    print("="*60)
    return True
