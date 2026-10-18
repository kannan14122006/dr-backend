"""Exploratory analysis of a DR dataset -> artifacts/dataset_stats.json (served at /api/dataset/stats).

Usage:
    python -m training.analyze_dataset --csv data/aptos/train.csv --images-dir data/aptos/train_images
"""
import argparse
import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from app.config import CLASS_NAMES, DATASET_STATS_PATH, REFERABLE_FROM_GRADE
from app.preprocess import crop_black_border
from training.data import load_labels


def thumbnail_data_url(path: Path, size: int = 220) -> str:
    with Image.open(path) as img:
        arr = crop_black_border(np.asarray(img.convert("RGB")))
    thumb = Image.fromarray(arr)
    thumb.thumbnail((size, size))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def describe(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "median": float(np.median(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--name", default=None, help="Dataset display name (default: CSV parent folder)")
    ap.add_argument("--sample", type=int, default=400, help="Images per run to measure size/brightness")
    ap.add_argument("--examples-per-class", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DATASET_STATS_PATH)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_labels(args.csv, args.images_dir)
    total = len(df)
    counts = df["label"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)

    classes = [
        {"grade": g, "label": CLASS_NAMES[g], "count": int(c), "percent": round(100 * c / total, 2)}
        for g, c in counts.items()
    ]
    nonzero = counts[counts > 0]
    referable = int(counts[counts.index >= REFERABLE_FROM_GRADE].sum())

    sample = df.sample(n=min(args.sample, total), random_state=args.seed)
    widths, heights, brightness = [], [], {g: [] for g in range(len(CLASS_NAMES))}
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Measuring images"):
        with Image.open(row["path"]) as img:
            widths.append(img.width)
            heights.append(img.height)
            small = img.convert("L")
            small.thumbnail((256, 256))
            brightness[row["label"]].append(float(np.asarray(small).mean()))

    examples = {}
    for g in range(len(CLASS_NAMES)):
        subset = df[df["label"] == g]
        picks = subset.sample(n=min(args.examples_per_class, len(subset)), random_state=args.seed)
        examples[str(g)] = [thumbnail_data_url(p) for p in picks["path"]]

    stats = {
        "dataset": args.name or args.csv.parent.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_images": total,
        "num_classes": len(CLASS_NAMES),
        "classes": classes,
        "imbalance_ratio": round(float(nonzero.max() / nonzero.min()), 2),
        "majority_class": CLASS_NAMES[int(counts.idxmax())],
        "minority_class": CLASS_NAMES[int(nonzero.idxmin())],
        "binary": {
            "non_referable": total - referable,
            "referable": referable,
            "referable_percent": round(100 * referable / total, 2),
            "referable_from_grade": REFERABLE_FROM_GRADE,
        },
        "image_size": {
            "sampled": len(sample),
            "width": describe(widths),
            "height": describe(heights),
            "unique_resolutions": len(set(zip(widths, heights))),
        },
        "mean_brightness_by_class": [
            {"grade": g, "label": CLASS_NAMES[g], "mean": round(float(np.mean(v)), 2) if v else None}
            for g, v in brightness.items()
        ],
        "examples": examples,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    for c in classes:
        print(f"  {c['grade']} {c['label']:<17} {c['count']:>6}  ({c['percent']}%)")
    print(f"  imbalance ratio {stats['imbalance_ratio']}x, referable {stats['binary']['referable_percent']}%")


if __name__ == "__main__":
    main()
