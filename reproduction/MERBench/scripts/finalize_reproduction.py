#!/usr/bin/env python
"""Finalize report/status files from actually generated results.

This script is intentionally evidence-driven: missing result rows leave the
corresponding experiment marked as pending instead of inventing values.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict[str, str]], name: str, split: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("name") == name and row.get("split") == split), None)


def fmt(value: str | None, digits: int = 2) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def update_report(root: Path, mult_row: dict[str, str] | None, multiseed: dict | None) -> None:
    path = root / "reports" / "merbench_report_draft.md"
    text = path.read_text(encoding="utf-8")
    if mult_row:
        local_mse = mult_row.get("local_mse", "") or "-"
        replacement = (
            f"| MulT high FRA | Table IX | 82.94 | - | "
            f"{fmt(mult_row.get('local_waf_percent'))} | "
            f"{fmt(mult_row.get('local_acc_percent'))} | "
            f"{fmt(mult_row.get('local_uar_percent'))} | {local_mse} |"
        )
        text = re.sub(r"\| MulT high FRA \| Table IX \|[^\n]+", replacement, text, count=1)
        text = text.replace(
            "MulT FRA 100 epoch 五折正式运行已经启动，目前已完成第 2 折并进入第 3 折；最终结果尚未生成前不写入本地数值。",
            "MulT FRA 100 epoch 五折正式运行已经完成，最终数值来自实际生成的 `.npz` 和五折最佳 checkpoint。",
        )
        text = text.replace(
            "MulT FRA 正式五折运行正在继续，不能在完成前把它写成最终结果。",
            "MulT FRA 正式五折运行已完成，结果已纳入本地总表并与论文 Table IX 分开列出。",
        )
    if multiseed and int(multiseed.get("n_runs", 0)) >= 3:
        text = text.replace(
            "论文表格通常报告多次运行均值和标准差，本地当前正式表主要是 seed 2026。",
            "Attention 已完成固定超参的多 seed 统计；LMF、MISA、TFN、MulT 等其他正式基线仍以 seed 2026 本地复现值为主，不能冒充论文均值。",
        )
        text = text.replace(
            "固定 Attention 的 seed 2027/2028 多 seed 重跑脚本已就绪，完成后与 seed 2026 一起计算均值和标准差。",
            "固定 Attention 的 seed 2026/2027/2028 多 seed 均值和标准差已生成，文件位于 `results/attention_multiseed_summary.*`。",
        )
    path.write_text(text, encoding="utf-8", newline="\n")


def update_status(root: Path, mult_row: dict[str, str] | None, multiseed: dict | None) -> None:
    path = root / "results" / "reproduction_status.json"
    if path.exists():
        status = json.loads(path.read_text(encoding="utf-8"))
    else:
        status = {}
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    completed = status.setdefault("completed_experiments", [])
    completed = [item for item in completed if item.get("name") != "MulT high FRA"]
    if mult_row:
        completed.append(
            {
                "name": "MulT high FRA",
                "type": "paper_baseline",
                "model": "mult",
                "result_dir": "results/mult_high_fra_seed2026_resume",
                "run_dir": "runs/mult_high_fra_seed2026_resume-trimodal",
                "log": "runs/mult_high_fra_seed2026_resume.log",
                "test1": {
                    "waf": float(mult_row["local_waf"]),
                    "acc": float(mult_row["local_acc"]),
                    "uar": float(mult_row["local_uar"]),
                    "mse": float(mult_row["local_mse"]),
                },
                "paper_waf": 82.94,
                "notes": "formal FRA run with fold-1 checkpoint restoration and batch size 16",
            }
        )
    status["completed_experiments"] = completed
    pending = list(status.get("pending", []))
    pending = [item for item in pending if "MulT" not in item and "multi-seed" not in item]
    if not mult_row:
        pending.append("MulT FRA 100 epoch formal run")
    if not multiseed or int(multiseed.get("n_runs", 0)) < 3:
        pending.append("multi-seed reruns")
    status["pending"] = pending
    status["limitations"] = [
        "Only high UTT/FRA features are available in the downloaded feature.zip.",
        "LMF, MISA, TFN, and MulT local formal results are single-seed values; only Attention has the fixed-hyperparameter three-seed summary.",
        "MISA default/random hyperparameters failed with NaN; stable hyperparameter rerun is reported separately.",
        "Top-k / medium / low feature experiments cannot be strictly reproduced without the corresponding official feature pool.",
        "Large pretrained feature extractors were not retrained or rerun.",
    ]
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = read_rows(root / "results" / "local_reproduction_summary.csv")
    mult_row = find_row(rows, "MulT high FRA", "test1")
    multiseed_path = root / "results" / "attention_multiseed_summary.json"
    multiseed = None
    if multiseed_path.exists():
        payload = json.loads(multiseed_path.read_text(encoding="utf-8"))
        multiseed = payload.get("aggregate", {})
    update_report(root, mult_row, multiseed)
    update_status(root, mult_row, multiseed)
    audit = {
        "mulT_test1_row_present": mult_row is not None,
        "attention_multiseed_runs": int(multiseed.get("n_runs", 0)) if multiseed else 0,
        "report_updated": True,
        "status_updated": True,
    }
    audit_path = root / "results" / "finalization_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
