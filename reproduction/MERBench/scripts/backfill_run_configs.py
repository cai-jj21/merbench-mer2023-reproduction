#!/usr/bin/env python
"""Recover run_config.json for older MERBench runs from their saved NPZ args."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs_root = root / "runs"
    written = 0
    skipped = 0
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "result").is_dir():
            continue
        config_path = run_dir / "run_config.json"
        if config_path.exists():
            skipped += 1
            continue
        result_files = sorted((run_dir / "result").glob("*.npz"))
        if not result_files:
            continue
        data = np.load(result_files[0], allow_pickle=True)
        if "args" not in data.files:
            continue
        saved_args = data["args"].item()
        if hasattr(saved_args, "__dict__"):
            config = dict(vars(saved_args))
        elif isinstance(saved_args, dict):
            config = dict(saved_args)
        else:
            config = {"saved_args": str(saved_args)}
        config["backfilled_from"] = str(result_files[0])
        config["checkpoint_count"] = len(list((run_dir / "model").glob("*.pt")))
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        written += 1
    print(json.dumps({"written": written, "skipped_existing": skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
