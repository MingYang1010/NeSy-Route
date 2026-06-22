# NeSy-Route

This repository contains the prompt templates and evaluation code for **NeSy-Route**, a neural-symbolic benchmark for constrained route planning in remote-sensing imagery.

<p align="center">
  <a href="https://mingyang1010.github.io/NeSy-Route/"><img src="https://img.shields.io/badge/Project-Page-2ea44f?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2603.16307"><img src="https://img.shields.io/badge/arXiv-2603.16307-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper"></a>
  <a href="https://github.com/MingYang1010/NeSy-Route"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Code"></a>
  <a href="https://huggingface.co/datasets/Ming1010/NeSy-Route"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000" alt="Hugging Face Dataset"></a>
</p>

<!--
## News

- 2026-06: NeSy-Route was accepted to ECCV 2026.
-->

## Repository Layout

```text
evaluation/
  vector_metrics.py                 # Shared Task 1/2 vector metrics
  task1/
    prompts.py                      # Task 1 prompt template
    evaluate.py                     # Traversability and cost-vector evaluation
  task2/
    prompts.py                      # Task 2 prompt template
    evaluate.py                     # Traversability, cost-vector, and region-vector evaluation
  task3/
    prompts.py                      # Task 3 prompt template
    trajectory_evaluator.py         # Task 3 CLI entrypoint
    trajectory_core.py              # Task 3 trajectory metrics
scripts/
  install_env.sh                    # Evaluation + vLLM/OpenAI-compatible API environment setup
requirements.txt                    # Evaluation dependencies
requirements-inference.txt          # vLLM and OpenAI API dependencies
```

Datasets, model outputs, generated metrics, and credentials are intentionally excluded from this repository.

## Installation

On a Linux machine with a CUDA-compatible GPU, run:

```bash
bash scripts/install_env.sh
source .venv/bin/activate
```

The script installs the evaluation dependencies plus `vllm`, `openai`, and `qwen-vl-utils`.

If you already have a managed environment, you can install manually:

```bash
pip install -r requirements.txt -r requirements-inference.txt
```

## Dataset

Download the dataset and set:

```bash
export NESY_ROUTE_DATA=/path/to/NeSy-Route
```

Expected layout:

```text
$NESY_ROUTE_DATA/
  Task1/
    task1_v4_filter_v2_updated.json
  Task2/
    Level_1.json
    Level_2.json
    Level_3.json
    Level_1/
    Level_2/
    Level_3/
  Task3/
    filter_fixed_total_with_id_xy.jsonl
    easy.jsonl
    medium.jsonl
    hard.jsonl
    filter_fixed_total_with_id_xy/
      samples/
      cost_map_npy/
      traverse_map_npy/
      gt/
```

Task 3 also needs semantic label masks. Pass their directory through `--labels_root`; label file names should match each sample's `image_name`.

## Running Inference

Prompt templates are provided in:

```text
evaluation/task1/prompts.py
evaluation/task2/prompts.py
evaluation/task3/prompts.py
```

For local inference, start an OpenAI-compatible vLLM server:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen2.5-vl-7b \
  --trust-remote-code
```

Then call it with the OpenAI Python client:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

For hosted APIs, set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and the model name according to your provider. Do not commit API keys or local credential files.

The evaluation scripts expect prediction files to follow these naming patterns:

```text
Task 1/2: <result_dir>/<model>_<dataset>_<prompt_version>.json
Task 3:   a JSONL model-output file passed through --model_output
```

## Task 1 Evaluation

Task 1 evaluates `pred_traverse_vector` and `pred_cost_vector`.

```bash
python evaluation/task1/evaluate.py \
  --model gpt-5.2 \
  --dataset task1_v4_filter_v2_updated \
  --prompt_version v1 \
  --dataset_dir "$NESY_ROUTE_DATA/Task1" \
  --result_dir results/task1 \
  --metrics_dir outputs/task1_metrics \
  --errors_dir outputs/task1_errors \
  --filtered_dir outputs/task1_filtered
```

Expected prediction item:

```json
{
  "source_id": 2,
  "success": true,
  "pred_traverse_vector": [1, 1, 1, 1, 0, 1, 1, 0],
  "pred_cost_vector": [2, 2, 2, 2, 0, 1, 2, 0]
}
```

## Task 2 Evaluation

Task 2 evaluates `pred_traverse_vector`, `pred_cost_vector`, and `pred_region_vector`.

```bash
python evaluation/task2/evaluate.py \
  --model gpt-5.1 \
  --dataset Level_1 \
  --prompt_version v1 \
  --dataset_dir "$NESY_ROUTE_DATA/Task2" \
  --result_dir results/task2 \
  --metrics_dir outputs/task2_metrics \
  --errors_dir outputs/task2_errors \
  --filtered_dir outputs/task2_filtered
```

Run the same command with `--dataset Level_2` and `--dataset Level_3` for the other levels.

Expected prediction item:

```json
{
  "id": 1,
  "success": true,
  "pred_traverse_vector": [0, 0, 0, 0, 0, 1, 0, 0],
  "pred_cost_vector": [0, 0, 0, 0, 0, 1, 0, 0],
  "pred_region_vector": [0, 0, 0, 0, 0, 1, 0, 0]
}
```

## Task 3 Evaluation

Task 3 evaluates predicted route waypoints or trajectories using traversability maps, cost maps, ground-truth paths, and semantic label masks.

Check paths first:

```bash
python evaluation/task3/trajectory_evaluator.py \
  --dataset "$NESY_ROUTE_DATA/Task3/filter_fixed_total_with_id_xy.jsonl" \
  --model_output results/task3/model_output.jsonl \
  --dataset_root "$NESY_ROUTE_DATA/Task3" \
  --labels_root /path/to/semantic/labels \
  --output_dir outputs/task3 \
  --check_paths
```

Run the full evaluation:

```bash
python evaluation/task3/trajectory_evaluator.py \
  --dataset "$NESY_ROUTE_DATA/Task3/filter_fixed_total_with_id_xy.jsonl" \
  --model_output results/task3/model_output.jsonl \
  --dataset_root "$NESY_ROUTE_DATA/Task3" \
  --labels_root /path/to/semantic/labels \
  --output_dir outputs/task3 \
  --num_workers 8
```

Run a difficulty subset:

```bash
python evaluation/task3/trajectory_evaluator.py \
  --dataset "$NESY_ROUTE_DATA/Task3/filter_fixed_total_with_id_xy.jsonl" \
  --model_output results/task3/model_output.jsonl \
  --dataset_root "$NESY_ROUTE_DATA/Task3" \
  --labels_root /path/to/semantic/labels \
  --output_dir outputs/task3 \
  --subset_jsonl "$NESY_ROUTE_DATA/Task3/easy.jsonl" \
  --num_workers 8
```
