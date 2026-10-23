"""Export the trained model to ONNX for in-browser inference (onnxruntime-web on GitHub Pages).

The exported graph contains the whole pipeline after crop + resize, so the browser only has to
crop the black border and resize to a square:

    input  "image"     float32 [1, 3, S, S]  RGB 0-255, cropped + resized fundus
    output "probs"     float32 [1, 5]        softmax over the 5 ICDR grades
    output "cams"      float32 [1, 5, h, w]  class activation maps (Grad-CAM equivalent for a GAP + linear head)
    output "processed" float32 [1, 3, S, S]  preprocessed image (local-average subtraction + disk mask)

Usage:
    python -m deploy.export_onnx --out ../dr-frontend/models --data-dir ../dr-frontend/data \
        --verify-csv <labels.csv> --verify-images <images dir>
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.config import CLASS_NAMES, DATASET_STATS_PATH, IMAGENET_MEAN, IMAGENET_STD, METRICS_PATH, MODEL_PATH
from app.model import build_model
from app.preprocess import prepare_fundus


class BrowserModel(nn.Module):
    def __init__(self, model: nn.Module, size: int, sigma_ratio: float = 30.0):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

        # Same Gaussian kernel OpenCV uses for GaussianBlur(ksize=(0,0), sigma) on uint8 images.
        sigma = size / sigma_ratio
        ksize = int(round(sigma * 6 + 1)) | 1
        k = torch.tensor(cv2.getGaussianKernel(ksize, sigma)[:, 0], dtype=torch.float32)
        self.pad = ksize // 2
        self.register_buffer("kx", k.view(1, 1, 1, -1).repeat(3, 1, 1, 1))
        self.register_buffer("ky", k.view(1, 1, -1, 1).repeat(3, 1, 1, 1))

        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (size // 2, size // 2), int(size * 0.5 * 0.95), 1, thickness=-1)
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32).view(1, 1, size, size))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        linear = model.classifier[-1]
        self.register_buffer("cam_weight", linear.weight.detach().clone()[:, :, None, None])

    def forward(self, image):
        p = self.pad
        blur = F.conv2d(F.pad(image, (p, p, 0, 0), mode="reflect"), self.kx, groups=3)
        blur = F.conv2d(F.pad(blur, (0, 0, p, p), mode="reflect"), self.ky, groups=3)
        blur = torch.round(blur)
        processed = torch.clamp(torch.round(4 * image - 4 * blur + 128), 0, 255)
        processed = processed * self.mask + 128 * (1 - self.mask)

        x = (processed / 255 - self.mean) / self.std
        feats = self.features(x)
        logits = self.classifier(torch.flatten(self.avgpool(feats), 1))
        cams = F.conv2d(feats, self.cam_weight)
        return torch.softmax(logits, dim=1), cams, processed


def to_nchw(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32))


def verify(onnx_path: Path, model: nn.Module, size: int, csv: Path, images: Path, n: int):
    """Compare ONNX (preprocessing in-graph) against the original Python/PyTorch pipeline."""
    from training.data import load_labels
    from app.model import to_tensor_transform

    df = load_labels(csv, images)
    sample = df.groupby("label", group_keys=False).apply(lambda g: g.sample(n=min(len(g), n // 5), random_state=0))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tf = to_tensor_transform()

    agree, max_prob_diff, pix_diffs = 0, 0.0, []
    for path in sample["path"]:
        with Image.open(path) as img:
            original, processed = prepare_fundus(img, size)
        with torch.no_grad():
            ref = torch.softmax(model(tf(Image.fromarray(processed)).unsqueeze(0)), 1)[0].numpy()
        probs, _, proc = sess.run(None, {"image": to_nchw(original)})
        agree += int(probs[0].argmax() == ref.argmax())
        max_prob_diff = max(max_prob_diff, float(np.abs(probs[0] - ref).max()))
        pix_diffs.append(float(np.abs(proc[0].transpose(1, 2, 0) - processed.astype(np.float32)).mean()))

    print(f"Verified on {len(sample)} images: grade agreement {agree}/{len(sample)}, "
          f"max probability diff {max_prob_diff:.4f}, mean preprocessed pixel diff {np.mean(pix_diffs):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    ap.add_argument("--out", type=Path, required=True, help="Folder for dr_model.onnx")
    ap.add_argument("--data-dir", type=Path, default=None, help="Folder to copy metrics/dataset JSON into")
    ap.add_argument("--verify-csv", type=Path, default=None)
    ap.add_argument("--verify-images", type=Path, default=None)
    ap.add_argument("--verify-n", type=int, default=50)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch, size = ckpt["arch"], ckpt["image_size"]
    if arch != "efficientnet_b0":
        raise SystemExit("Browser export currently supports efficientnet_b0 checkpoints")
    model = build_model(arch, len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "dr_model.onnx"
    torch.onnx.export(
        BrowserModel(model, size).eval(),
        (torch.full((1, 3, size, size), 128.0),),
        str(onnx_path),
        input_names=["image"],
        output_names=["probs", "cams", "processed"],
        opset_version=17,
        dynamo=False,
    )
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 2**20:.1f} MB)")

    info = {
        "arch": arch, "image_size": size, "classes": ckpt.get("class_names", CLASS_NAMES),
        "epoch": ckpt.get("epoch"), "val_kappa": ckpt.get("val_kappa"), "referable_from_grade": 2,
    }
    (args.out / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if args.data_dir:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        for src in (METRICS_PATH, DATASET_STATS_PATH):
            if src.exists():
                shutil.copy(src, args.data_dir / src.name)
                print(f"Copied {src.name} -> {args.data_dir}")

    if args.verify_csv and args.verify_images:
        verify(onnx_path, model, size, args.verify_csv, args.verify_images, args.verify_n)


if __name__ == "__main__":
    main()
