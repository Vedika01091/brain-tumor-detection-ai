# Brain Tumor Detection with AI

A web application that classifies a brain MRI slice as **glioma**, **meningioma**, **pituitary tumour** or
**no tumour**, shows the confidence for every class, and overlays a **Grad-CAM heatmap** that highlights the
region the model relied on.

- **Model:** EfficientNet-B0 (PyTorch), fine-tuned on the Kaggle *Brain Tumor MRI* dataset.
- **Back end:** Flask REST API (`backend/`). **Front end:** React 19 + Vite (`frontend/`).

> Academic prototype. It is **not** a medical device and has not been validated on clinical data.

## Model performance

Measured on the 1,311 held-out test images of the Kaggle dataset (5,712 were used for training/validation):

| Class | Recall |
|---|---|
| Glioma | 99.7% |
| Meningioma | 100% |
| No tumour | 100% |
| Pituitary | 99.7% |
| **Overall accuracy** | **99.85%** (2 errors) |

A prediction with heatmap takes about 0.15 s on a laptop CPU. The dataset may contain very similar slices on both
sides of its train/test split, so accuracy on scans from a new hospital is likely to be lower. The dataset and the
training scripts are not part of this copy of the project; the full details are in the project report.

## Project layout

```
backend/                          Flask API (app.py) and PyTorch inference + Grad-CAM (efficientnet_infer.py)
frontend/                         React (Vite) interface
model/efficientnet_v2/            Trained model: checkpoint (.pth), class list, evaluation report
report/                           Project report (.docx and .pdf)
```

## Setup

You need Python 3.10+ and Node.js 18+.

```
python -m venv venv
venv/bin/pip install -r backend/requirements.txt          # Windows: venv\Scripts\pip install -r backend\requirements.txt
# optional, for a smaller CPU-only PyTorch download, install torch and torchvision first:
#   venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

cd frontend
npm install
```

## Run

Start the API (terminal 1), then the interface (terminal 2):

```
venv/bin/python backend/app.py          # http://localhost:5000     (Windows: venv\Scripts\python backend\app.py)

cd frontend
npm run dev                             # http://localhost:5173
```

Open http://localhost:5173, upload an MRI image (PNG, JPG or BMP) and press **Analyze Scan**. Use **Original** /
**Grad-CAM Heatmap** to switch views. The Vite dev server proxies `/api/*` to the Flask API, so no CORS setup is needed.

To build the interface for production: `cd frontend && npm run build` (output in `frontend/dist/`).

## API

| Route | Description |
|---|---|
| `GET /api/health` | Liveness, and whether the model is loaded |
| `GET /api/model-info` | Class names and the saved evaluation metrics |
| `POST /api/predict` | Multipart field `image` (max 16 MB). Returns `prediction`, `tumor_detected`, `confidence`, `probabilities`, `original_image` and `heatmap_overlay` (base64 PNG data URIs) |

Invalid input returns a JSON error: 400 (missing file, unsupported type, unreadable image), 413 (above 16 MB),
503 (model not loaded).

```
curl -F "image=@scan.jpg" http://localhost:5000/api/predict
```

## Configuration

| Environment variable | Meaning | Default |
|---|---|---|
| `EFFICIENTNET_CHECKPOINT` | Use another checkpoint: a `.pth` file or an extracted `.pth` folder (`data.pkl` + `data/`) | `model/efficientnet_v2/brain_tumor_efficientnet_v2.pth` |
| `BACKEND_HOST` / `BACKEND_PORT` | Where the API listens | `127.0.0.1` / `5000` |
| `FLASK_DEBUG` | `1` enables Flask debug mode (development only) | off |

A replacement checkpoint must be an EfficientNet-B0 with a 4-class head, trained with this pre-processing:
brain-region crop, resize to 224x224, ImageNet normalisation. The class order is in
`model/efficientnet_v2/class_indices.json`.

## Notes

- Uploaded images are processed in memory and not stored.
- Inference is serialised with a lock because the Grad-CAM hooks keep state on the shared model; concurrent requests
  are queued. The built-in Flask server is meant for development; use a production WSGI server and add authentication
  before any public deployment.
- A non-MRI image is not rejected; it still receives one of the four labels.

## Report

`report/Brain_Tumor_Detection_Project_Report.docx` (editable) and `.pdf`.
