#!/usr/bin/env python
"""Build compact MERBench reproduction summary tables from per-run CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    model: str
    modalities: str
    feature_set: str
    result_dir: str
    prefix: str
    run_dir: str | None
    log_file: str | None
    paper_waf: float | None = None
    paper_mse: float | None = None
    notes: str = ""


EXPERIMENTS = [
    Experiment(
        "Attention high",
        "paper_baseline",
        "Attention",
        "audio+text+visual",
        "HUBERT-large + Baichuan-13B + CLIP-large",
        "attention_high_seed2026_ckpt",
        "attention_high_seed2026_ckpt",
        "attention_high_seed2026_ckpt-trimodal",
        "attention_high_seed2026_ckpt.log",
        paper_waf=83.03,
        paper_mse=0.62,
        notes="checkpoint rerun; paper WAF from Table IX, MSE from Table III HL+BA+CL",
    ),
    Experiment(
        "LMF high",
        "paper_baseline",
        "LMF",
        "audio+text+visual",
        "HUBERT-large + Baichuan-13B + CLIP-large",
        "lmf_high_seed2026",
        "lmf_high_seed2026",
        "lmf_high_seed2026-trimodal",
        "lmf_high_seed2026.log",
        paper_waf=83.44,
        paper_mse=None,
        notes="completed before checkpoint patch; predictions and logs saved",
    ),
    Experiment(
        "MISA high (default)",
        "paper_baseline",
        "MISA",
        "audio+text+visual",
        "HUBERT-large + Baichuan-13B + CLIP-large",
        "misa_high_seed2026",
        "misa_high_seed2026",
        "misa_high_seed2026-trimodal",
        "misa_high_seed2026.log",
        paper_waf=82.42,
        paper_mse=None,
        notes="official UTT fusion baseline from Table IX; local default run failed with NaN and is kept as failure evidence",
    ),
    Experiment(
        "MISA high (stable)",
        "paper_baseline",
        "MISA",
        "audio+text+visual",
        "HUBERT-large + Baichuan-13B + CLIP-large",
        "misa_high_stable_seed2026",
        "misa_high_stable_seed2026",
        "misa_high_stable_seed2026-trimodal",
        "misa_high_stable_seed2026.log",
        paper_waf=82.42,
        paper_mse=None,
        notes="stable local rerun after default MISA produced NaN; uses configs/hyper_stable.yaml",
    ),
    Experiment(
        "TFN high",
        "paper_baseline",
        "TFN",
        "audio+text+visual",
        "HUBERT-large + Baichuan-13B + CLIP-large",
        "tfn_high_seed2026",
        "tfn_high_seed2026",
        "tfn_high_seed2026-trimodal",
        "tfn_high_seed2026.log",
        paper_waf=82.52,
        paper_mse=None,
        notes="official UTT fusion baseline from Table IX",
    ),
    Experiment(
        "MulT high FRA",
        "paper_baseline",
        "MulT",
        "audio+text+visual",
        "HUBERT-large-FRA + Baichuan-13B-FRA + CLIP-large-FRA",
        "mult_high_fra_seed2026_resume",
        "mult_high_fra_seed2026_resume",
        "mult_high_fra_seed2026_resume-trimodal",
        "mult_high_fra_seed2026_resume.log",
        paper_waf=82.94,
        paper_mse=None,
        notes="official frame-aligned fusion baseline from Table IX; resumed local run uses fixed configs/hyper_stable.yaml and batch size 16 for 8GB VRAM",
    ),
    Experiment(
        "Audio only",
        "unimodal",
        "Attention",
        "audio",
        "HUBERT-large",
        "unimodal_audio_attention_seed2026",
        "unimodal_audio_attention_seed2026",
        "unimodal_audio_attention_seed2026-unimodal",
        "unimodal_audio_attention_seed2026.log",
        paper_waf=69.91,
        paper_mse=0.97,
        notes="completed before checkpoint patch; predictions and logs saved",
    ),
    Experiment(
        "Text only",
        "unimodal",
        "Attention",
        "text",
        "Baichuan-13B",
        "unimodal_text_attention_seed2026",
        "unimodal_text_attention_seed2026",
        "unimodal_text_attention_seed2026-unimodal",
        "unimodal_text_attention_seed2026.log",
        paper_waf=62.54,
        paper_mse=1.63,
        notes="completed before checkpoint patch; validation was unstable",
    ),
    Experiment(
        "Visual only",
        "unimodal",
        "Attention",
        "visual",
        "CLIP-large",
        "unimodal_visual_attention_seed2026",
        "unimodal_visual_attention_seed2026",
        "unimodal_visual_attention_seed2026-unimodal",
        "unimodal_visual_attention_seed2026.log",
        paper_waf=62.69,
        paper_mse=1.12,
        notes="completed before checkpoint patch; predictions and logs saved",
    ),
    Experiment(
        "Attention + class weight",
        "improvement",
        "Attention",
        "audio+text+visual",
        "HUBERT-large + Baichuan-13B + CLIP-large",
        "attention_high_classweight_seed2026_ckpt",
        "attention_high_classweight_seed2026_ckpt",
        "attention_high_classweight_seed2026_ckpt-trimodal",
        "attention_high_classweight_seed2026_ckpt.log",
        paper_waf=None,
        paper_mse=None,
        notes="local improvement: balanced CE class weights",
    ),
    Experiment(
        "Gated residual Attention",
        "improvement",
        "GatedResidualAttention",
        "audio+text+visual",
        "HUBERT-large + Baichuan-13B + CLIP-large",
        "gated_residual_attention_high_seed2026_formal",
        "gated_residual_attention_high_seed2026_formal",
        "gated_residual_attention_high_seed2026_formal-trimodal",
        "gated_residual_attention_high_seed2026_formal.log",
        paper_waf=None,
        paper_mse=None,
        notes="local structural improvement: normalized modality reliability gates plus cross-modal residual branch",
    ),
    Experiment(
        "Drop audio",
        "ablation",
        "Attention",
        "text+visual",
        "HUBERT-large + Baichuan-13B + CLIP-large",
        "attention_high_drop_audio_seed2026",
        "attention_high_drop_audio_seed2026",
        "attention_high_drop_audio_seed2026-trimodal",
        "attention_high_drop_audio_seed2026.log",
        paper_waf=None,
        paper_mse=None,
        notes="modality missing ablation; audio feature zeroed after alignment",
    ),
    Experiment(
        "Drop text",
        "ablation",
        "Attention",
        "audio+visual",
        "HUBERT-large + Baichuan-13B + CLIP-large",
        "attention_high_drop_text_seed2026",
        "attention_high_drop_text_seed2026",
        "attention_high_drop_text_seed2026-trimodal",
        "attention_high_drop_text_seed2026.log",
        paper_waf=None,
        paper_mse=None,
        notes="modality missing ablation; text feature zeroed after alignment",
    ),
    Experiment(
        "Drop visual",
        "ablation",
        "Attention",
        "audio+text",
        "HUBERT-large + Baichuan-13B + CLIP-large",
        "attention_high_drop_visual_seed2026",
        "attention_high_drop_visual_seed2026",
        "attention_high_drop_visual_seed2026-trimodal",
        "attention_high_drop_visual_seed2026.log",
        paper_waf=None,
        paper_mse=None,
        notes="modality missing ablation; visual feature zeroed after alignment",
    ),
]


def read_split(csv_path: Path, split: str) -> dict[str, str] | None:
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == split:
                return row
    return None


def checkpoint_count(root: Path, run_dir: str | None) -> int:
    if not run_dir:
        return 0
    model_dir = root / "runs" / run_dir / "model"
    if not model_dir.exists():
        return 0
    return len(list(model_dir.glob("*.pt")))


def fmt_float(value: str | float | None, ndigits: int = 4) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{ndigits}f}"


def to_percent(value: str | float | None) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value) * 100:.2f}"


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    headers = [
        "group",
        "name",
        "split",
        "local_waf_percent",
        "local_acc_percent",
        "local_uar_percent",
        "local_mse",
        "paper_waf_percent",
        "paper_mse",
        "checkpoint_count",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row.get(h, "") for h in headers) + " |\n")


def build(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for exp in EXPERIMENTS:
        csv_path = root / "results" / exp.result_dir / f"{exp.prefix}.csv"
        ckpts = checkpoint_count(root, exp.run_dir)
        for split in ["cv", "test1", "test2", "test3"]:
            row = read_split(csv_path, split)
            if row is None:
                continue
            rows.append(
                {
                    "group": exp.group,
                    "name": exp.name,
                    "model": exp.model,
                    "modalities": exp.modalities,
                    "feature_set": exp.feature_set,
                    "split": split,
                    "n_samples": row.get("n_samples", ""),
                    "local_waf": fmt_float(row.get("waf") or row.get("filename_waf")),
                    "local_acc": fmt_float(row.get("accuracy") or row.get("filename_acc")),
                    "local_uar": fmt_float(row.get("uar")),
                    "local_mse": fmt_float(
                        row.get("val_mse_valid_labels_only")
                        or (row.get("val_mse") if str(row.get("val_mse_comparable", "")).lower() != "false" else "")
                        or (row.get("filename_val_mse") if str(row.get("val_mse_comparable", "")).lower() != "false" else "")
                    ),
                    "local_waf_percent": to_percent(row.get("waf") or row.get("filename_waf")),
                    "local_acc_percent": to_percent(row.get("accuracy") or row.get("filename_acc")),
                    "local_uar_percent": to_percent(row.get("uar")),
                    "val_missing_count": row.get("val_missing_count", ""),
                    "val_mse_comparable": row.get("val_mse_comparable", ""),
                    "paper_waf_percent": fmt_float(exp.paper_waf, 2),
                    "paper_mse": fmt_float(exp.paper_mse, 2),
                    "checkpoint_count": str(ckpts),
                    "run_dir": exp.run_dir or "",
                    "log_file": exp.log_file or "",
                    "notes": exp.notes,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="reproduction/MERBench root",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build(root)
    csv_path = out_dir / "local_reproduction_summary.csv"
    json_path = out_dir / "local_reproduction_summary.json"
    md_path = out_dir / "local_reproduction_summary.md"

    fieldnames = [
        "group",
        "name",
        "model",
        "modalities",
        "feature_set",
        "split",
        "n_samples",
        "local_waf",
        "local_acc",
        "local_uar",
        "local_mse",
        "local_waf_percent",
        "local_acc_percent",
        "local_uar_percent",
        "paper_waf_percent",
        "paper_mse",
        "val_missing_count",
        "val_mse_comparable",
        "checkpoint_count",
        "run_dir",
        "log_file",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    write_markdown(rows, md_path)

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
