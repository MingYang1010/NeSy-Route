<div align="center">

<p>
  <a href="https://eccv.ecva.net/Conferences/2026">
    <img src="assets/eccv-2026-logo.svg" alt="ECCV 2026" width="180">
  </a>
</p>

<h1>NeSy-Route</h1>

<p><strong>A neuro-symbolic benchmark for constrained route planning in remote-sensing imagery.</strong></p>

<p><strong>🎉 NeSy-Route has been accepted to ECCV 2026!</strong></p>

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
| 2026.07 | 🎉 **NeSy-Route has been accepted to ECCV 2026.** |
| 2026.07 | 🚀 Evaluation-ready annotations, Task 3 semantic masks, and evaluation code are released. |
| 2026.06 | 🌍 The project page, paper, code, and benchmark dataset are online. |

## Overview

NeSy-Route evaluates whether multimodal large language models can combine visual perception, symbolic constraints, and route planning over remote-sensing imagery.

| Task | Setting | Samples | Evaluation target |
| --- | --- | ---: | --- |
| Task 1 | Few-shot | 3,607 | Textual traversability and terrain-preference reasoning |
| Task 2 | Zero-shot | 12,975 | Text-image constraint alignment and region recognition |
| Task 3 | Zero-shot | 10,821 | Constraint-aware waypoint and trajectory planning |

Prompt templates are provided under `evaluation/task1`, `evaluation/task2`, and `evaluation/task3`.

## Installation

Create an evaluation environment:

```bash
git clone https://github.com/MingYang1010/NeSy-Route.git
cd NeSy-Route
bash scripts/install_env.sh
source .venv/bin/activate
```

To additionally install vLLM and OpenAI-compatible inference dependencies:

```bash
bash scripts/install_env.sh --with-inference
source .venv/bin/activate
```

## Prepare Evaluation Data

The benchmark images remain in the main Hugging Face Parquet configurations. Scoring only needs the lightweight ground-truth annotations and the deduplicated Task 3 semantic masks:

```bash
python scripts/download_evaluation_data.py --task all
```

This creates `data/NeSy-Route/` and downloads approximately 110 MiB rather than the complete image release. A single task or difficulty can also be selected:

```bash
python scripts/download_evaluation_data.py --task task2 --split easy
python scripts/download_evaluation_data.py --task task3 --split hard
```

For model inference, load the image-bearing configurations directly:

```python
from datasets import load_dataset

task2_easy = load_dataset("Ming1010/NeSy-Route", "task2", split="easy")
task3_easy = load_dataset("Ming1010/NeSy-Route", "task3", split="easy")
```

## Prediction Format

Save one JSON object per line. The scorer accepts the field names emitted by the provided prompts.

Task 1:

```json
{"query_id":"Q00001","traverse_vector":[1,1,1,1,0,1,1,0],"cost_vector":[2,2,2,2,0,1,2,0]}
```

Task 2:

```json
{"sample_id":"easy_000001","traverse_vector":[0,0,0,0,0,1,0,0],"cost_vector":[0,0,0,0,0,1,0,0],"region_vector":[0,0,0,0,0,1,0,0]}
```

Task 3 uses integer pixel coordinates in `[x, y]` order:

```json
{"sample_id":"easy_000572","trajectory":[[108,104],[118,114],[128,124],[617,688]]}
```

The legacy `pred_traverse_vector`, `pred_cost_vector`, `pred_region_vector`, and `id` aliases remain supported.

## Run Evaluation

Task 1 reports Traversability Matching (TM), Preference Ranking Correlation (PR), and Fully Matching Accuracy (FM):

```bash
python evaluate.py task1 \
  --predictions results/task1.jsonl
```

Task 2 reports Region Matching Rate (RM), TM, and PR for one difficulty split:

```bash
python evaluate.py task2 \
  --split easy \
  --predictions results/task2_easy.jsonl
```

Task 3 reports Adherence Rate (AR), Cost Ratio (CR), Violation Ratio (VR), and Chamfer Distance (CD):

```bash
python evaluate.py task3 \
  --split easy \
  --predictions results/task3_easy.jsonl \
  --num-workers 8 \
  --strict
```

Task 3 stores each semantic mask once on Hugging Face. During evaluation, the scorer combines the mask with each sample's traversability and cost vectors to reconstruct the query-specific maps used by AR, CR, and VR. The first run materializes a local mask cache under `data/NeSy-Route/Task3/.label_cache/`.

Without `--strict`, a prediction subset can be scored for development. Use `--strict` for complete benchmark reporting.

## OpenAI-Compatible Inference

The prompts can be used with hosted APIs or a local vLLM server. For example:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen2.5-vl-7b \
  --trust-remote-code

export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

For hosted providers, set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and the model name according to the provider. Do not commit credentials or raw API responses containing secrets.

## Citation

```bibtex
@inproceedings{yang2026nesyroute,
  title     = {NeSy-Route: A Neuro-Symbolic Benchmark for Constrained Route Planning in Remote Sensing},
  author    = {Yang, Ming and Zhou, Zhi and Tian, Shiyu and Yu, Kunyang and Guo, Lan-Zhe and Li, Yu-Feng},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```
