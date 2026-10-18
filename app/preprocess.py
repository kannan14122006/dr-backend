"""Fundus image preprocessing.

Mirrors the steps reported in the survey (Papadopoulos et al. [65], Quellec et al. [40]):
  1. crop the black border around the retina disk,
  2. resize to a square input,
  3. subtract the local average colour (Gaussian blur) to normalise lighting,
  4. zero out the outer ~5% of the retina disk.
"""
import cv2
import numpy as np
from PIL import Image


def crop_black_border(img: np.ndarray, tol: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def subtract_local_average(img: np.ndarray, sigma_ratio: float = 30.0) -> np.ndarray:
    size = img.shape[0]
    blur = cv2.GaussianBlur(img, (0, 0), size / sigma_ratio)
    out = cv2.addWeighted(img, 4, blur, -4, 128)

    # Keep only the inner 95% of the retina disk.
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * 0.5 * 0.95), 1, thickness=-1)
    out = out * mask[..., None] + 128 * (1 - mask[..., None])
    return out.astype(np.uint8)


def prepare_fundus(pil_img: Image.Image, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (cropped_resized_rgb, model_input_rgb), both uint8 HxWx3."""
    img = np.asarray(pil_img.convert("RGB"))
    img = crop_black_border(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img, subtract_local_average(img)
