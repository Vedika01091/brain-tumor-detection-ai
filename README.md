# Brain Tumor Detection with AI

4-class brain MRI classifier (glioma / meningioma / pituitary / no tumour)
served through a Flask API with live Grad-CAM heatmaps, and a React frontend
for uploading scans and reviewing results, against the Kaggle Brain MRI
dataset already in `dataset/`.

Two interchangeable model backends sit behind the same API contract:

- **efficientnet** (default) - PyTorch EfficientNet-B0, full fine-tune with
  focal loss, class weighting, brain-region cropping and OneCycleLR.
  Trained by `new model/main.py`. Artifacts in `model/efficientnet/`.
- **vgg16** - the original two-stage VGG16 model (conv feature extractor +
  dense classifier head). Trained by `ml/train.py`. Artifacts in `model/`.
  Implements the methodology from `ilovepdf_merged (2) (1).pdf` (Ch. 5-6).

Switch backends with the `MODEL_BACKEND` env var (`efficientnet` or `vgg16`)
before starting the Flask server - no code or frontend changes needed.

## Layout

```
dataset/                Kaggle Brain MRI dataset (Training/ and Testing/, 4 classes)
new model/main.py         EfficientNet-B0 training pipeline -> model/efficientnet/
new model/gradcam.py       Standalone Grad-CAM report-figure generator (PyTorch)
new model/confusion_matrix_main.py  Standalone eval/confusion-matrix generator (PyTorch)
ml/train.py               VGG16 training pipeline: bottleneck features -> head -> fine-tune -> eval
ml/gradcam.py              Grad-CAM heatmap utility (TensorFlow; shared by train.py and the backend)
ml/evaluate_api.py         Integration test: drives the live Flask API across the whole test set
model/                     VGG16 artifacts + evaluation report/plots (generated)
model/efficientnet/         EfficientNet-B0 artifacts + evaluation report/plots (generated)
backend/app.py             Flask API: /api/health, /api/model-info, /api/predict (dual backend)
backend/efficientnet_infer.py  PyTorch inference + Grad-CAM used by the Flask API
frontend/                   React (Vite) UI: upload, diagnosis, Grad-CAM overlay, history
venv/                        Python environment (TensorFlow, PyTorch, Flask, etc.)
```

## 1. Train a model

EfficientNet-B0 (default backend):

```
venv\Scripts\python.exe "new model\main.py"
```

Produces in `model/efficientnet/`: `brain_tumor_efficientnet_best.pth`,
`class_indices.json`, `evaluation_report.json`, `confusion_matrix.png`,
`training_curves.png`, `gradcam_examples.png`. On CPU, full fine-tuning for
up to 20 epochs (early stopping, patience 5) takes roughly 2-3 hours.

VGG16 (legacy backend):

```
venv\Scripts\python.exe ml\train.py
```

Produces in `model/`: `vgg16_base.keras`, `classifier_head.keras`,
`class_indices.json`, `evaluation_report.json`, `confusion_matrix.png`,
`training_curves.png`, `gradcam_examples.png`. On CPU this takes roughly
15-45+ minutes depending on hardware (VGG16 fine-tuning is the slow part).

## 2. Run the backend

```
venv\Scripts\python.exe backend\app.py
```

Serves the API on `http://localhost:5000` using the EfficientNet backend by
default. Set `MODEL_BACKEND=vgg16` to serve the legacy model instead. Works
even before training finishes (`/api/health` reports `model_loaded: false`
until the active backend's files exist).

## 3. Run the frontend

```
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173`; its dev server proxies `/api/*` to the
Flask backend, so no CORS config is needed on your end.

## Notes

- This is an academic prototype, not a medical device — the UI carries an
  explicit disclaimer and this is not intended for clinical use.
- Re-running `new model\main.py` overwrites the files in `model/efficientnet/`;
  re-running `ml/train.py` overwrites the files in `model/`. The two
  backends never touch each other's artifacts.
- `ml/evaluate_api.py` drives the live API across the whole test set and
  writes `api_evaluation_report.json` next to whichever backend's
  `evaluation_report.json` the running server reports via `/api/health`.
