"""FastAPI service: DR grading of fundus images + dataset/model analysis endpoints."""
import base64
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from PIL import Image, UnidentifiedImageError

from .config import (
    ALLOWED_ORIGINS,
    CLASS_NAMES,
    DATASET_STATS_PATH,
    FRONTEND_URL,
    MAX_UPLOAD_BYTES,
    METRICS_PATH,
    MODEL_PATH,
    REFERABLE_FROM_GRADE,
)
from .model import Predictor

log = logging.getLogger("dr-backend")
state: dict = {"predictor": None, "load_error": None, "loaded_at": None}
STARTED_AT = time.time()


def load_predictor() -> None:
    if not MODEL_PATH.exists():
        state["predictor"] = None
        state["load_error"] = f"No trained model at {MODEL_PATH}. Run `python -m training.train` first."
        log.warning(state["load_error"])
        return
    try:
        state["predictor"] = Predictor(MODEL_PATH)
        state["load_error"] = None
        state["loaded_at"] = datetime.now(timezone.utc).isoformat()
        log.info("Loaded %s model from %s", state["predictor"].arch, MODEL_PATH)
    except Exception as exc:  # surface the reason through /api/health
        state["predictor"] = None
        state["load_error"] = f"Failed to load model: {exc}"
        log.exception("Model load failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_predictor()
    yield


API_DESCRIPTION = """
Grades retinal fundus photographs on the **5-level ICDR diabetic retinopathy scale**:

| Grade | Stage | Referable |
|---|---|---|
| 0 | No DR | no |
| 1 | Mild NPDR | no |
| 2 | Moderate NPDR | **yes** |
| 3 | Severe NPDR | **yes** |
| 4 | Proliferative DR | **yes** |

Every prediction returns the probability of all five grades, a referable-DR verdict,
and a **Grad-CAM** heatmap that shows which retina regions drove the decision.

*This is an academic project and not a medical device.*
"""

TAGS = [
    {"name": "Prediction", "description": "Grade a fundus image."},
    {"name": "Model", "description": "Model status, validation metrics and hot reload."},
    {"name": "Dataset", "description": "Exploratory analysis of the training dataset."},
    {"name": "Service", "description": "Health check and landing page."},
]

app = FastAPI(
    title="RetinaScan DR API",
    summary="Deep-learning diabetic retinopathy grading service",
    description=API_DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS,
    lifespan=lifespan,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1, "docExpansion": "list", "syntaxHighlight.theme": "monokai"},
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_data_url(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def read_json(path, missing_hint: str) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=missing_hint)
    return json.loads(path.read_text(encoding="utf-8"))



LANDING_PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<circle cx="16" cy="16" r="14" fill="#c2410c"/><circle cx="20" cy="13" r="4" fill="#fde68a"/></svg>'
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page():
    return LANDING_PAGE.replace("{{FRONTEND_URL}}", FRONTEND_URL)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(FAVICON, media_type="image/svg+xml")


@app.get("/api/health", tags=["Service"], summary="Service and model status")
def health():
    p = state["predictor"]
    return {
        "status": "ok",
        "version": app.version,
        "uptime_seconds": round(time.time() - STARTED_AT),
        "model_loaded": p is not None,
        "model_error": state["load_error"],
        "model_loaded_at": state["loaded_at"],
        "arch": p.arch if p else None,
        "image_size": p.image_size if p else None,
        "checkpoint_epoch": p.epoch if p else None,
        "checkpoint_val_kappa": p.val_kappa if p else None,
        "device": str(p.device) if p else None,
        "classes": CLASS_NAMES,
        "referable_from_grade": REFERABLE_FROM_GRADE,
    }


@app.post("/api/model/reload", tags=["Model"], summary="Reload the checkpoint from disk after retraining")
def reload_model():
    load_predictor()
    return health()


@app.get("/api/model/metrics", tags=["Model"], summary="Validation metrics and training history")
def model_metrics():
    return read_json(METRICS_PATH, "No training metrics yet. Run `python -m training.train`.")


@app.get("/api/dataset/stats", tags=["Dataset"], summary="Class distribution, image stats and samples")
def dataset_stats():
    return read_json(DATASET_STATS_PATH, "Dataset not analysed yet. Run `python -m training.analyze_dataset`.")


@app.post(
    "/api/predict",
    tags=["Prediction"],
    summary="Grade a fundus image (0 No DR to 4 Proliferative DR)",
    responses={
        400: {"description": "File is not a readable image"},
        413: {"description": "Image too large"},
        503: {"description": "No trained model available"},
    },
)
async def predict(file: UploadFile = File(..., description="Retinal fundus photograph (PNG/JPEG/TIFF)")):
    predictor: Predictor | None = state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail=state["load_error"] or "Model not loaded")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image")

    started = time.perf_counter()
    result = predictor.predict(img)
    result["images"] = {k: to_data_url(v) for k, v in result["images"].items()}
    result["filename"] = file.filename
    result["arch"] = predictor.arch
    result["inference_ms"] = round((time.perf_counter() - started) * 1000)
    return result
