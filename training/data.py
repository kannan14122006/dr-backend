"""Dataset loading for APTOS 2019 / EyePACS style CSV labels."""
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from app.preprocess import prepare_fundus

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# (id column, label column) for the common public DR datasets.
KNOWN_SCHEMAS = [
    ("id_code", "diagnosis"),  # APTOS 2019
    ("image", "level"),  # EyePACS / Kaggle 2015
    ("Image name", "Retinopathy grade"),  # IDRiD
    ("image_id", "label"),
]


def load_labels(csv_path: Path, images_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with columns [id, label, path] for images that exist on disk."""
    df = pd.read_csv(csv_path)
    for id_col, label_col in KNOWN_SCHEMAS:
        if id_col in df.columns and label_col in df.columns:
            break
    else:
        raise ValueError(f"Unrecognised CSV columns {list(df.columns)}; expected one of {KNOWN_SCHEMAS}")

    files = {p.stem: p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS}
    df = pd.DataFrame({"id": df[id_col].astype(str).str.replace(r"\.\w+$", "", regex=True), "label": df[label_col]})
    df["path"] = df["id"].map(files)

    missing = df["path"].isna().sum()
    if missing:
        print(f"[warn] {missing} labelled images not found in {images_dir}; skipping them")
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    if df.empty:
        raise ValueError(f"No images from {csv_path} found in {images_dir}")
    return df


def cap_per_class(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Randomly undersample classes larger than max_per_class (e.g. the dominant No DR grade)."""
    parts = [g.sample(n=min(len(g), max_per_class), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


class FundusDataset(Dataset):
    """Fundus images -> preprocessed tensors.

    With cache_dir set, each preprocessed image is written once as PNG and reused in later
    epochs, which makes CPU training far faster than re-decoding full-size JPEGs every epoch.
    """

    def __init__(self, df: pd.DataFrame, image_size: int, transform, cache_dir: Path | None = None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.image_size = image_size
        self.transform = transform
        self.cache_dir = Path(cache_dir) / str(image_size) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def _processed(self, path: Path) -> Image.Image:
        cached = self.cache_dir / f"{Path(path).stem}.png" if self.cache_dir else None
        if cached and cached.exists():
            try:
                with Image.open(cached) as img:
                    return img.convert("RGB")
            except OSError:
                pass  # partially written file; regenerate it
        with Image.open(path) as img:
            _, processed = prepare_fundus(img, self.image_size)
        out = Image.fromarray(processed)
        if cached:
            tmp = cached.with_suffix(f".{os.getpid()}.tmp")
            out.save(tmp, format="PNG")
            os.replace(tmp, cached)
        return out

    def __getitem__(self, idx):
        return self.transform(self._processed(self.paths[idx])), self.labels[idx]
