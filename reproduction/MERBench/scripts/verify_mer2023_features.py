import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_FEATURES = [
    "chinese-hubert-large-UTT",
    "Baichuan-13B-Base-UTT",
    "clip-vit-large-patch14-UTT",
]


def load_label_file(label_path: Path) -> dict:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    data = np.load(label_path, allow_pickle=True)
    return {key: data[key].tolist() for key in data.files}


def split_summary(corpus: dict) -> dict:
    emotions = Counter(item.get("emo") for item in corpus.values())
    missing_valence = sum(
        item.get("val", "") == "" or item.get("val") is None for item in corpus.values()
    )
    placeholder_valence = sum(item.get("val") == -10 for item in corpus.values())
    return {
        "samples": len(corpus),
        "emotions": dict(emotions),
        "missing_valence": missing_valence,
        "placeholder_valence": placeholder_valence,
    }


def read_feature(feature_root: Path, sample_name: str):
    npy_path = feature_root / f"{sample_name}.npy"
    dir_path = feature_root / sample_name
    if npy_path.exists():
        arr = np.load(npy_path).squeeze()
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr
    if dir_path.is_dir():
        arrays = [np.load(path) for path in sorted(dir_path.glob("*.npy"))]
        arr = np.array(arrays).squeeze()
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr
    return None


def inspect_feature(data_root: Path, feature_name: str, sample_names: list[str]) -> dict:
    feature_root = data_root / "features" / feature_name
    result = {
        "feature": feature_name,
        "root": str(feature_root),
        "exists": feature_root.exists(),
        "checked_samples": len(sample_names),
        "missing_samples": [],
        "example_shape": None,
    }
    if not feature_root.exists():
        result["missing_samples"] = sample_names
        return result

    for name in sample_names:
        arr = read_feature(feature_root, name)
        if arr is None:
            result["missing_samples"].append(name)
        elif result["example_shape"] is None:
            result["example_shape"] = list(arr.shape)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument("--samples-per-split", type=int, default=5)
    args = parser.parse_args()

    data_root = args.data_root
    label_path = data_root / "label-6way.npz"
    corpora = load_label_file(label_path)

    summary = {
        "data_root": str(data_root),
        "label_path": str(label_path),
        "splits": {key: split_summary(value) for key, value in corpora.items()},
        "features": [],
    }

    sample_names = []
    for split_name in ["train_corpus", "test1_corpus", "test2_corpus", "test3_corpus"]:
        corpus = corpora.get(split_name, {})
        sample_names.extend(list(corpus.keys())[: args.samples_per_split])

    for feature_name in args.features:
        summary["features"].append(inspect_feature(data_root, feature_name, sample_names))

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    missing_roots = [item["feature"] for item in summary["features"] if not item["exists"]]
    missing_samples = [
        item["feature"] for item in summary["features"] if item["missing_samples"]
    ]
    if missing_roots or missing_samples:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

