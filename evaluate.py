#!/usr/bin/env python3
"""Dispatch a NeSy-Route task evaluator through one public entrypoint."""

from __future__ import annotations

import runpy
import sys


TASK_MODULES = {
    "task1": "evaluation.task1.evaluate",
    "task2": "evaluation.task2.evaluate",
    "task3": "evaluation.task3.trajectory_evaluator",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in TASK_MODULES:
        choices = " | ".join(TASK_MODULES)
        raise SystemExit(f"Usage: python evaluate.py <{choices}> [task arguments]")
    task = sys.argv.pop(1)
    runpy.run_module(TASK_MODULES[task], run_name="__main__")


if __name__ == "__main__":
    main()
