"""Reject images that are not retinal fundus photographs (screenshots, documents, ordinary photos).

Fundus photos have a bright, orange-red retina disk on a dark background inside a roughly
square crop. Thresholds were set on EyePACS: on 3,000 held-out fundus images 1.1% were
flagged (mostly very dark or washed-out images that are ungradable anyway), and 0 of 38
non-fundus images (screenshots, an Excel sheet, documents, wallpapers, noise) passed.
Mirrored in dr-frontend/js/inference.js -- keep both in sync.
"""
import numpy as np
from PIL import Image

ANALYSIS_SIZE = 256

REASONS = {
    "shape": "The image isn't the round, roughly square shape of a fundus photo.",
    "no_dark_border": "No dark border around a round retina disk was found.",
    "colour": "The colours don't match a retina (expected orange-red tones).",
    "too_dark": "The retina area is too dark to grade (poor image quality).",
}


def fundus_features(pil_img: Image.Image) -> dict | None:
    img = pil_img.convert("RGB")
    img.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE))
    a = np.asarray(img).astype(np.int64)
    gray = (a[..., 0] * 4899 + a[..., 1] * 9617 + a[..., 2] * 1868 + 8192) >> 14
    mask = gray > 7
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    crop = a[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    g = gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    ch, cw = g.shape

    # Disk of diameter = crop width, centred in the crop (fundus disks may be clipped top/bottom).
    yy, xx = np.mgrid[0:ch, 0:cw]
    inside = (xx + 0.5 - cw / 2) ** 2 + (yy + 0.5 - ch / 2) ** 2 <= (cw / 2) ** 2
    outside = ~inside
    bright = inside & (g > 25)
    r, gr, b = crop[bright].mean(axis=0) if bright.any() else (0.0, 0.0, 0.0)

    return {
        "aspect": cw / ch,
        "border_dark": float((g[outside] <= 25).mean()) if outside.sum() > 50 else 1.0,
        "disk_fill": float(bright.sum() / max(1, inside.sum())),
        "red_share": float(r / max(1.0, r + gr + b)),
        "r_gt_g": float(r - gr),
        "g_gt_b": float(gr - b),
    }


def fundus_problems(pil_img: Image.Image) -> list[str]:
    """Return reason codes why this doesn't look like a gradable fundus photo ([] = looks fine)."""
    f = fundus_features(pil_img)
    if f is None:
        return ["too_dark"]
    problems = []
    if not 0.9 <= f["aspect"] <= 1.55:
        problems.append("shape")
    if f["border_dark"] < 0.8:
        problems.append("no_dark_border")
    if f["r_gt_g"] < -5 or f["g_gt_b"] < -6 or f["red_share"] < 0.33:
        problems.append("colour")
    if f["disk_fill"] < 0.15:
        problems.append("too_dark")
    return problems
