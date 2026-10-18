"""Central configuration for the DR classification backend."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.getenv("DR_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))
MODEL_PATH = Path(os.getenv("DR_MODEL_PATH", ARTIFACTS_DIR / "model.pt"))
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
DATASET_STATS_PATH = ARTIFACTS_DIR / "dataset_stats.json"

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5500")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024

# International Clinical Diabetic Retinopathy (ICDR) severity scale,
# the 5-level grading used by APTOS 2019 / EyePACS (see survey Fig. 1).
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NUM_CLASSES = len(CLASS_NAMES)

# Grades >= 2 (moderate or worse) are "referable DR" (survey section IV-A).
REFERABLE_FROM_GRADE = 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
