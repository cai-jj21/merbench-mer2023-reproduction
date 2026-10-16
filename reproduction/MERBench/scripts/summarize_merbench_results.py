import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    recall_score,
)


EMOTION_NAMES = ["neutral", "angry", "happy", "sad", "worried", "surprise"]
METRIC_RE = re.compile(r"(?:^|_)f1_(?P<f1>\d+(?:\.\d+)?)_acc_(?P<acc>\d+(?:\.\d+)?)_val_(?P<val>\d+(?:\.\d+)?)_")


def parse_filename_metrics(path):
    match = METRIC_RE.search(path.stem)
    if not match:
        return {}
    return {key: float(value) for key, value in match.groupdict().items()}


def as_float_array(values):
    try:
        return np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([])


def write_confusion_csv(path, matrix):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", *EMOTION_NAMES])
        for label, row in zip(EMOTION_NAMES, matrix):
            writer.writerow([label, *row.tolist()])


def summarize_npz(path, out_dir):
    row = {
        "file": str(path),
        "split": path.name.split("_", 1)[0],
        "has_predictions": False,
    }
    metrics = parse_filename_metrics(path)
    row.update(
        {
            "filename_waf": metrics.get("f1", ""),
            "filename_acc": metrics.get("acc", ""),
            "filename_val_mse": metrics.get("val", ""),
        }
    )

    data = np.load(path, allow_pickle=True)
    if {"emolabels", "emopreds"}.issubset(set(data.files)):
        labels = np.asarray(data["emolabels"], dtype=int).reshape(-1)
        preds = np.asarray(data["emopreds"], dtype=int).reshape(-1)
        row["has_predictions"] = True
        row["n_samples"] = int(labels.shape[0])
        row["waf"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
        row["accuracy"] = float(accuracy_score(labels, preds))
        row["uar"] = float(recall_score(labels, preds, average="macro", zero_division=0))

        stem = path.stem
        report = classification_report(
            labels,
            preds,
            labels=list(range(len(EMOTION_NAMES))),
            target_names=EMOTION_NAMES,
            output_dict=True,
            zero_division=0,
        )
        report_path = out_dir / f"{stem}_classification_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        row["classification_report"] = str(report_path)

        matrix = confusion_matrix(labels, preds, labels=list(range(len(EMOTION_NAMES))))
        matrix_path = out_dir / f"{stem}_confusion_matrix.csv"
        write_confusion_csv(matrix_path, matrix)
        row["confusion_matrix"] = str(matrix_path)

    if {"vallabels", "valpreds"}.issubset(set(data.files)):
        val_labels = as_float_array(data["vallabels"])
        val_preds = as_float_array(data["valpreds"])
        if val_labels.size and val_labels.shape == val_preds.shape:
            missing = val_labels <= -9.0
            row["val_missing_count"] = int(missing.sum())
            row["val_mse_comparable"] = bool(not missing.any())
            if missing.any():
                row["val_mse_all_including_placeholders"] = float(mean_squared_error(val_labels, val_preds))
                valid = ~missing
                if valid.any():
                    row["val_mse_valid_labels_only"] = float(mean_squared_error(val_labels[valid], val_preds[valid]))
            else:
                row["val_mse"] = float(mean_squared_error(val_labels, val_preds))
                row["val_mse_valid_labels_only"] = row["val_mse"]

    return row


def write_summary_csv(rows, path):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(rows, path):
    columns = ["split", "filename_waf", "filename_acc", "filename_val_mse", "waf", "accuracy", "uar", "n_samples"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, help="Directory containing MERBench .npz result files.")
    parser.add_argument("--out-dir", required=True, help="Directory for CSV/Markdown/report outputs.")
    parser.add_argument("--prefix", default="merbench_results", help="Output filename prefix.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(result_dir.glob("*.npz"))
    rows = [summarize_npz(path, out_dir) for path in files]

    csv_path = out_dir / f"{args.prefix}.csv"
    md_path = out_dir / f"{args.prefix}.md"
    json_path = out_dir / f"{args.prefix}.json"
    write_summary_csv(rows, csv_path)
    write_summary_md(rows, md_path)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
