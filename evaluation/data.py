"""I/O helpers for the public NeSy-Route release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


HF_DATASET_REPO = "Ming1010/NeSy-Route"
SPLITS = ("easy", "medium", "hard")


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        records = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "predictions", "matches"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported JSON record format: {input_path}")


def load_predictions(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        return {}, load_json_records(input_path)

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {}, data
    if isinstance(data, dict):
        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        for key in ("results", "predictions"):
            if isinstance(data.get(key), list):
                return meta, data[key]
    raise ValueError(f"Unsupported prediction format: {input_path}")


def parquet_files(path: str | Path) -> list[Path]:
    input_path = Path(path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"No Parquet files found at {input_path}")


def load_parquet_records(
    path: str | Path,
    columns: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    selected = list(columns) if columns is not None else None
    tables = [pq.read_table(file_path, columns=selected) for file_path in parquet_files(path)]
    return pa.concat_tables(tables, promote_options="default").to_pylist()


def save_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def successful(row: dict[str, Any]) -> bool:
    return row.get("success", True) is not False


def require_file(path: str | Path, hint: str | None = None) -> Path:
    file_path = Path(path)
    if file_path.is_file():
        return file_path
    message = f"Required file not found: {file_path}"
    if hint:
        message += f"\n{hint}"
    raise FileNotFoundError(message)
