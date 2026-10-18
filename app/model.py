"""Model definition, inference and Grad-CAM explainability."""
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, REFERABLE_FROM_GRADE
from .preprocess import prepare_fundus

SUPPORTED_ARCHS = ("resnet50", "efficientnet_b0")


def build_model(arch: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ImageNet pre-trained backbone with a new 5-class head (transfer learning)."""
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from {SUPPORTED_ARCHS}")
    return model


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    return model.layer4 if arch == "resnet50" else model.features[-1]


def to_tensor_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class GradCAM:
    """Class activation map of the predicted grade (which retina regions drove the decision)."""

    def __init__(self, layer: nn.Module):
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad.detach()

    def heatmap(self, size: int) -> np.ndarray:
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cv2.resize(cam.cpu().numpy(), (size, size))


class Predictor:
    def __init__(self, checkpoint_path: Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.arch = ckpt["arch"]
        self.image_size = ckpt["image_size"]
        self.class_names = ckpt.get("class_names", CLASS_NAMES)
        self.epoch = ckpt.get("epoch")
        self.val_kappa = ckpt.get("val_kappa")
        self.model = build_model(self.arch, len(self.class_names), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.cam = GradCAM(gradcam_layer(self.model, self.arch))
        self.transform = to_tensor_transform()
        self._lock = threading.Lock()

    def predict(self, pil_img: Image.Image) -> dict:
        original, processed = prepare_fundus(pil_img, self.image_size)
        x = self.transform(Image.fromarray(processed)).unsqueeze(0).to(self.device)

        with self._lock, torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            grade = int(probs.argmax())
            logits[0, grade].backward()
            heat = self.cam.heatmap(self.image_size)

        probs = probs.detach().cpu().numpy()
        colored = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[..., ::-1]
        overlay = np.uint8(0.55 * original + 0.45 * colored)

        return {
            "grade": grade,
            "label": self.class_names[grade],
            "confidence": float(probs[grade]),
            "probabilities": [
                {"grade": i, "label": name, "probability": float(p)}
                for i, (name, p) in enumerate(zip(self.class_names, probs))
            ],
            "expected_grade": float(np.dot(probs, np.arange(len(probs)))),
            "referable": grade >= REFERABLE_FROM_GRADE,
            "referable_probability": float(probs[REFERABLE_FROM_GRADE:].sum()),
            "images": {"original": original, "processed": processed, "gradcam": overlay},
        }
