<div align="center">

<h1>NeSy-Route</h1>

<p><strong>A neural-symbolic benchmark for constrained route planning in remote-sensing imagery.</strong></p>

<p>
  <a href="https://mingyang1010.github.io/NeSy-Route/">
    <img src="https://cdn.simpleicons.org/googlechrome/34A853" alt="Project Page" height="18">
    <b>Project Page</b>
  </a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2603.16307">
    <img src="https://cdn.simpleicons.org/arxiv/B31B1B" alt="arXiv" height="18">
    <b>Paper</b>
  </a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://github.com/MingYang1010/NeSy-Route">
    <img src="https://cdn.simpleicons.org/github/181717" alt="GitHub" height="18">
    <b>Code</b>
  </a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/Ming1010/NeSy-Route">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" height="18">
    <b>Dataset</b>
  </a>
</p>

</div>

## ✨ News

| Date | Update |
| --- | --- |
| 2026.06 | 🚀 Evaluation code is released, and the Hugging Face dataset repository is online. |
| 2026.06 | 🌍 Project page, paper, code, and dataset links are collected above for quick access. |

## Overview

This repository provides the prompt templates and evaluation scripts for **NeSy-Route**, a benchmark for studying whether multimodal large language models can combine visual perception, symbolic constraints, and route planning over remote-sensing imagery.

NeSy-Route contains three evaluation tasks:

- **Task 1:** few-shot semantic traversability and cost-vector prediction.
- **Task 2:** zero-shot constraint-aware semantic and region reasoning.
- **Task 3:** zero-shot constrained route planning with predicted waypoints or trajectories.

## Quick Start

Install the evaluation and inference dependencies:

```bash
bash scripts/install_env.sh
source .venv/bin/activate
```

If you already have a managed Python environment, install the dependencies manually:

```bash
pip install -r requirements.txt -r requirements-inference.txt
```

Download the dataset from Hugging Face and set:

```bash
export NESY_ROUTE_DATA=/path/to/NeSy-Route
```

Task 3 also needs semantic label masks. Pass the label directory through `--labels_root`; label file names should match each sample's `image_name`.

## Prompts and Predictions

Prompt templates are provided in:

```text
evaluation/task1/prompts.py
evaluation/task2/prompts.py
evaluation/task3/prompts.py
```

The evaluation scripts expect prediction files to follow these naming patterns:

```text
Task 1/2: <result_dir>/<model>_<dataset>_<prompt_version>.json
Task 3:   a JSONL model-output file passed through --model_output
```

For local inference, you can use an OpenAI-compatible vLLM server:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen2.5-vl-7b \
  --trust-remote-code
```

Then configure the OpenAI client:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

For hosted APIs, set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and the model name according to your provider. Do not commit API keys or credential files.

## Evaluation

### Task 1

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

### Task 2

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

Run the same command with `--dataset Level_2` and `--dataset Level_3` for the other difficulty levels.

### Task 3

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

Evaluate a difficulty subset:

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
