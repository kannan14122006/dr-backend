# Diabetic Retinopathy Classifier: Backend

A FastAPI and PyTorch service that grades a retinal fundus image on the 5-level ICDR scale:

| Grade | Stage | Referable? |
|---|---|---|
| 0 | No DR | No |
| 1 | Mild NPDR | No |
| 2 | Moderate NPDR | **Yes** |
| 3 | Severe NPDR | **Yes** |
| 4 | Proliferative DR | **Yes** |

The approach follows *Atwany et al., "Deep Learning Techniques for Diabetic Retinopathy Classification: A Survey", IEEE Access 2022*:

- **Transfer learning**: an ImageNet-pretrained ResNet50 (or EfficientNet-B0) with a new 5-class head.
- **Preprocessing**: the black border is cropped, the image is resized, the local average colour is subtracted (Gaussian blur), and the outer 5% of the retina disk is masked.
- **Augmentation**: flips, rotation, and brightness/contrast jitter.
- **Class imbalance**: inverse-frequency class weights. No DR makes up about 49% of APTOS.
- **Metrics**: accuracy, quadratic weighted kappa, macro F1, one-vs-rest AUC, a confusion matrix, and referable-DR sensitivity, specificity and AUC.
- **Explainability**: Grad-CAM heatmaps show which retina regions drove each prediction.

The web frontend lives in a separate repository: https://github.com/kannan14122006/dr-frontend

## Project layout

```
app/
  config.py         paths, class names, settings
  preprocess.py     fundus preprocessing
  model.py          model builder, inference, Grad-CAM
  main.py           FastAPI app
training/
  data.py           CSV and image loading (APTOS / EyePACS / IDRiD formats)
  analyze_dataset.py  dataset EDA -> artifacts/dataset_stats.json
  train.py          training -> artifacts/model.pt + artifacts/metrics.json
artifacts/          generated outputs (served by the API)
```

## 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/ first. Training on CPU works, but it is slow.

## 2a. Dataset used in this project: EyePACS resized (Downloads/archive)

This is the Kaggle "Diabetic Retinopathy Detection" set (EyePACS 2015), resized to 1024px. It has 35,126 labelled images, matching survey Table 1:

| Grade | Images | Share |
|---|---|---|
| 0 No DR | 25,810 | 73.5% |
| 1 Mild | 2,443 | 7.0% |
| 2 Moderate | 5,292 | 15.1% |
| 3 Severe | 873 | 2.5% |
| 4 Proliferative | 708 | 2.0% |

```
C:\Users\admin\Downloads\archive\trainLabels.csv                      # image,level
C:\Users\admin\Downloads\archive\resized_train\resized_train\*.jpeg
```

The commands used:

```bash
python -m training.analyze_dataset --csv C:\Users\admin\Downloads\archive\trainLabels.csv --images-dir C:\Users\admin\Downloads\archive\resized_train\resized_train --name "EyePACS (Kaggle DR 2015, resized)"

# CPU run: balanced subset of up to 1,500 images per grade (~6k images), EfficientNet-B0 @ 224px
python -m training.train --csv C:\Users\admin\Downloads\archive\trainLabels.csv --images-dir C:\Users\admin\Downloads\archive\resized_train\resized_train --name "EyePACS (Kaggle DR 2015)" --arch efficientnet_b0 --image-size 224 --epochs 12 --batch-size 32 --lr 5e-4 --max-per-class 1500 --workers 4
```

On a CUDA GPU, drop `--max-per-class` to train on all 35k images, and consider `--arch resnet50 --image-size 384`. Preprocessed images are cached in `data/cache/<size>/`, so every epoch after the first is much faster.

## 2b. Alternative dataset (APTOS 2019)

Download it from https://www.kaggle.com/competitions/aptos2019-blindness-detection/data and extract it so you have:

```
data/aptos/train.csv            # columns: id_code, diagnosis (0-4)
data/aptos/train_images/*.png   # 3,662 fundus images
```

EyePACS (`image,level`) and IDRiD CSVs work as well.

## 3. Analyse the dataset

```bash
python -m training.analyze_dataset --csv data/aptos/train.csv --images-dir data/aptos/train_images --name "APTOS 2019"
```

This writes `artifacts/dataset_stats.json`. It contains the class distribution, imbalance ratio, referable split, image resolution stats, brightness per grade and sample images per grade. The frontend's **Dataset Analysis** page shows it.

## 4. Train the model

```bash
python -m training.train --csv data/aptos/train.csv --images-dir data/aptos/train_images --epochs 15
```

Useful flags:

- `--arch efficientnet_b0`: a lighter model.
- `--image-size 512`: a larger input size.
- `--batch-size 8`: use this if you run out of GPU memory.
- `--limit 200 --epochs 1`: a quick smoke test.
- `--workers 0`: use this if Windows DataLoader workers give you trouble.

Training keeps the checkpoint with the best validation quadratic kappa in `artifacts/model.pt` and writes `artifacts/metrics.json`, which the **Model Performance** page shows.

## 5. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Or double-click `run.bat`.

- http://localhost:8000 is the landing page. It shows live service status, model metrics, training progress, the dataset summary, the endpoint list and copy-paste code samples.
- http://localhost:8000/docs is the interactive Swagger UI, where you can upload an image and try `/api/predict`.
- http://localhost:8000/redoc is the API reference.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Shows whether a model is loaded, plus the architecture and device |
| POST | `/api/predict` | multipart `file`: returns the grade, label, confidence, all 5 probabilities, referable flag, and the original, preprocessed and Grad-CAM images |
| GET | `/api/dataset/stats` | Dataset analysis JSON |
| GET | `/api/model/metrics` | Training and validation metrics JSON |
| POST | `/api/model/reload` | Reloads `artifacts/model.pt` after retraining |

Environment variables: `ALLOWED_ORIGINS` (CORS, default `*`), `DR_MODEL_PATH`, `DR_ARTIFACTS_DIR`, `MAX_UPLOAD_MB`.

> **Disclaimer:** This is an academic mini project. It is not a medical device and must not be used for clinical diagnosis.
