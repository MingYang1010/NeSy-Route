"""Command line entrypoint for Task 3 trajectory evaluation."""

import argparse

try:
    from .trajectory_core import batch_evaluate, check_data_paths
except ImportError:  # pragma: no cover - supports direct script execution.
    from trajectory_core import batch_evaluate, check_data_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formal trajectory evaluator for NeSy-Route Task 3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/task3/trajectory_evaluator.py \
    --dataset /path/to/NeSy-Route/Task3/filter_fixed_total_with_id_xy.jsonl \
    --model_output results/task3/model_output.jsonl \
    --dataset_root /path/to/NeSy-Route/Task3 \
    --labels_root /path/to/openearthmap/labels \
    --output_dir outputs/task3 \
    --num_workers 8

  python evaluation/task3/trajectory_evaluator.py \
    --dataset /path/to/NeSy-Route/Task3/filter_fixed_total_with_id_xy.jsonl \
    --model_output results/task3/model_output.jsonl \
    --dataset_root /path/to/NeSy-Route/Task3 \
    --labels_root /path/to/openearthmap/labels \
    --output_dir outputs/task3 \
    --subset_jsonl /path/to/NeSy-Route/Task3/easy.jsonl

  python evaluation/task3/trajectory_evaluator.py \
    --dataset /path/to/NeSy-Route/Task3/filter_fixed_total_with_id_xy.jsonl \
    --model_output results/task3/model_output.jsonl \
    --dataset_root /path/to/NeSy-Route/Task3 \
    --labels_root /path/to/openearthmap/labels \
    --output_dir outputs/task3 \
    --check_paths
        """,
    )

    parser.add_argument("--dataset", type=str, required=True, help="Task 3 dataset JSONL file.")
    parser.add_argument("--model_output", type=str, required=True, help="Model output JSONL file.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Task 3 root containing the dataset-specific folder, e.g. Task3/.",
    )
    parser.add_argument(
        "--labels_root",
        type=str,
        required=True,
        help="Semantic label mask folder; files should match image_name with .tif extension.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for evaluation outputs.")
    parser.add_argument("--num_workers", type=int, default=None, help="Parallel workers; default uses CPU count.")
    parser.add_argument("--check_paths", action="store_true", help="Validate paths without running evaluation.")
    parser.add_argument("--subset_jsonl", type=str, default=None, help="Optional difficulty subset JSONL.")
    args = parser.parse_args()

    if args.check_paths:
        check_data_paths(
            dataset_path=args.dataset,
            model_output_path=args.model_output,
            dataset_root=args.dataset_root,
            labels_root=args.labels_root,
        )
        return

    batch_evaluate(
        dataset_path=args.dataset,
        model_output_path=args.model_output,
        dataset_root=args.dataset_root,
        labels_root=args.labels_root,
        output_dir=args.output_dir,
        subset_jsonl=args.subset_jsonl,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
