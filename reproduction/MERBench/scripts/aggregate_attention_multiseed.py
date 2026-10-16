#!/usr/bin/env python
"""Aggregate fixed-hyperparameter Attention runs on MER-MULTI/test1."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_test1(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "test1":
                return row
    return None


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result_root = root / "results"
    runs = {
        2026: "attention_high_seed2026_ckpt",
        2027: "attention_high_seed2027_stable",
        2028: "attention_high_seed2028_stable",
    }
    rows: list[dict[str, object]] = []
    for seed, name in runs.items():
        row = read_test1(result_root / name / f"{name}.csv")
        if row is None:
            continue
        rows.append(
            {
                "seed": seed,
                "run": name,
                "waf_percent": float(row["waf"]) * 100,
                "accuracy_percent": float(row["accuracy"]) * 100,
                "uar_percent": float(row["uar"]) * 100,
                "mse": float(row.get("val_mse_valid_labels_only") or row.get("val_mse")),
                "n_samples": int(row["n_samples"]),
            }
        )
    if not rows:
        raise SystemExit("no completed fixed-hyperparameter Attention runs found")

    metric_names = ["waf_percent", "accuracy_percent", "uar_percent", "mse"]
    aggregate: dict[str, object] = {
        "experiment": "Attention fixed hyperparameters",
        "split": "MER-MULTI/test1",
        "seeds": [row["seed"] for row in rows],
        "n_runs": len(rows),
    }
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        mean, std = mean_std(values)
        aggregate[f"{metric}_mean"] = mean
        aggregate[f"{metric}_std"] = std

    csv_path = result_root / "attention_multiseed_summary.csv"
    json_path = result_root / "attention_multiseed_summary.json"
    md_path = result_root / "attention_multiseed_summary.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({})
        writer.writerow({"run": "mean", **{metric: aggregate[f"{metric}_mean"] for metric in metric_names}})
        writer.writerow({"run": "std", **{metric: aggregate[f"{metric}_std"] for metric in metric_names}})
    json_path.write_text(json.dumps({"runs": rows, "aggregate": aggregate}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Attention 多 seed 汇总",
        "",
        "固定超参：`configs/attention_stable.yaml`；主指标：MER-MULTI/test1。",
        "",
        "| seed | WAF (%) | ACC (%) | UAR (%) | MSE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['waf_percent']:.2f} | {row['accuracy_percent']:.2f} | {row['uar_percent']:.2f} | {row['mse']:.4f} |")
    lines.append(f"| mean +/- std | {aggregate['waf_percent_mean']:.2f} +/- {aggregate['waf_percent_std']:.2f} | {aggregate['accuracy_percent_mean']:.2f} +/- {aggregate['accuracy_percent_std']:.2f} | {aggregate['uar_percent_mean']:.2f} +/- {aggregate['uar_percent_std']:.2f} | {aggregate['mse_mean']:.4f} +/- {aggregate['mse_std']:.4f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
