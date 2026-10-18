"""Train a 5-level DR grading model with transfer learning.

Usage:
    python -m training.train --csv data/aptos/train.csv --images-dir data/aptos/train_images

Writes artifacts/model.pt (best validation quadratic kappa) and artifacts/metrics.json.
"""
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from app.config import ARTIFACTS_DIR, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, REFERABLE_FROM_GRADE
from app.model import SUPPORTED_ARCHS, build_model, to_tensor_transform
from training.data import FundusDataset, cap_per_class, load_labels


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_transform():
    # Flips / rotation / colour jitter: the augmentations most surveyed papers use against overfitting.
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180, fill=128),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def safe_auc(y_true, scores, **kw):
    try:
        return float(roc_auc_score(y_true, scores, **kw))
    except ValueError:  # a class missing from the validation split
        return None


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Multi-class + referable-DR binary metrics (survey Table 2: Acc, SN, SP, AUC, F1, Kappa)."""
    labels = list(range(len(CLASS_NAMES)))
    y_pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    yb_true = (y_true >= REFERABLE_FROM_GRADE).astype(int)
    yb_pred = (y_pred >= REFERABLE_FROM_GRADE).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb_true, yb_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auc_ovr": safe_auc(y_true, probs, multi_class="ovr", average="macro", labels=labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": [
            {"grade": g, "label": CLASS_NAMES[g], "precision": float(p[g]), "recall": float(r[g]),
             "f1": float(f[g]), "support": int(s[g])}
            for g in labels
        ],
        "referable": {
            "accuracy": float((tp + tn) / len(y_true)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "auc": safe_auc(yb_true, probs[:, REFERABLE_FROM_GRADE:].sum(1)),
        },
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    all_probs, all_labels, total_loss = [], [], 0.0
    for x, y in tqdm(loader, desc="  val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        all_probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    y_true, probs = np.concatenate(all_labels), np.concatenate(all_probs)
    return total_loss / len(y_true), y_true, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--arch", default="resnet50", choices=SUPPORTED_ARCHS)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Use only N images (quick smoke test)")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="Undersample any grade above N images (speeds up CPU training on EyePACS)")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"),
                    help="Where preprocessed images are cached ('' to disable)")
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--name", default=None, help="Dataset display name")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_labels(args.csv, args.images_dir)
    if args.max_per_class:
        df = cap_per_class(df, args.max_per_class, args.seed)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    train_df, val_df = train_test_split(df, test_size=args.val_split, stratify=stratify, random_state=args.seed)
    print(f"Train {len(train_df)} / Val {len(val_df)} images")
    print("Class counts:", df["label"].value_counts().sort_index().to_dict())

    cache = args.cache_dir if str(args.cache_dir) else None
    persistent = args.workers > 0
    train_loader = DataLoader(
        FundusDataset(train_df, args.image_size, train_transform(), cache),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=len(train_df) > args.batch_size,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        FundusDataset(val_df, args.image_size, to_tensor_transform(), cache),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        persistent_workers=persistent,
    )

    # Inverse-frequency class weights to counter the heavy No-DR imbalance.
    counts = np.bincount(train_df["label"], minlength=len(CLASS_NAMES)).astype(float)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                    label_smoothing=0.05)

    model = build_model(args.arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history, best_kappa, best = [], -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)

        val_loss, y_true, probs = evaluate(model, val_loader, device, criterion)
        m = compute_metrics(y_true, probs)
        history.append({
            "epoch": epoch, "train_loss": running / seen, "val_loss": val_loss,
            "val_accuracy": m["accuracy"], "val_kappa": m["quadratic_kappa"], "val_macro_f1": m["macro_f1"],
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  train_loss {running / seen:.4f}  val_loss {val_loss:.4f}  "
              f"acc {m['accuracy']:.4f}  kappa {m['quadratic_kappa']:.4f}  f1 {m['macro_f1']:.4f}")

        if m["quadratic_kappa"] > best_kappa:
            best_kappa, best = m["quadratic_kappa"], {**m, "epoch": epoch}
            torch.save({
                "arch": args.arch, "image_size": args.image_size, "class_names": CLASS_NAMES,
                "state_dict": model.state_dict(), "epoch": epoch, "val_kappa": best_kappa,
            }, args.out_dir / "model.pt")
            print(f"  saved new best model (kappa {best_kappa:.4f})")

        (args.out_dir / "metrics.json").write_text(json.dumps({
            "arch": args.arch, "image_size": args.image_size, "dataset": args.name or args.csv.parent.name, "max_per_class": args.max_per_class,
            "train_images": len(train_df), "val_images": len(val_df),
            "epochs": args.epochs, "trained_at": datetime.now(timezone.utc).isoformat(),
            "best": best, "history": history,
        }, indent=2), encoding="utf-8")

    print(f"Done. Best val kappa {best_kappa:.4f} at epoch {best['epoch']}. Artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
