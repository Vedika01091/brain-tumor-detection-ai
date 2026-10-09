"""Flask API for the Brain Tumor Detection system.

Supports two interchangeable model backends behind the same API contract
(selected via the MODEL_BACKEND env var; "efficientnet" is the default):

  - "efficientnet": PyTorch EfficientNet-B0, full fine-tune, trained by
    "new model/main.py". Checkpoint + metrics live in model/efficientnet/.
  - "vgg16": the original two-stage VGG16 model (conv feature extractor +
    dense classifier head) produced by ml/train.py. Checkpoint + metrics
    live in model/. Kept working, but no longer the default.

Routes:
  GET  /api/health        - liveness + whether the trained model is loaded
  GET  /api/model-info     - class list + saved evaluation metrics (if any)
  POST /api/predict        - multipart image upload -> prediction + Grad-CAM

Run with:  venv/Scripts/python.exe backend/app.py
"""
import base64
import io
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
EFFICIENTNET_DIR = os.path.join(MODEL_DIR, "efficientnet")
sys.path.insert(0, os.path.join(BASE_DIR, "ml"))

IMG_SIZE = (224, 224)
DISPLAY_MAX_SIDE = 512
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "efficientnet").strip().lower()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
CORS(app)

_state = {
    "backend": MODEL_BACKEND,
    "loaded": False,
    "load_error": None,
    "class_names": None,
    # "vgg16" backend (TensorFlow)
    "base_model": None,
    "head_model": None,
    "grad_model": None,
    "preprocess_input": None,
    # "efficientnet" backend (PyTorch)
    "torch_model": None,
    "gradcam": None,
}


def _load_vgg16_model():
    import tensorflow as tf  # imported lazily so /api/health responds fast even mid-import
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from gradcam import build_grad_model, compute_gradcam

    base_path = os.path.join(MODEL_DIR, "vgg16_base.keras")
    head_path = os.path.join(MODEL_DIR, "classifier_head.keras")
    classes_path = os.path.join(MODEL_DIR, "class_indices.json")

    if not (os.path.exists(base_path) and os.path.exists(head_path)):
        _state["load_error"] = (
            "Trained VGG16 model files not found in model/. Run ml/train.py first."
        )
        return

    base_model = tf.keras.models.load_model(base_path)
    head_model = tf.keras.models.load_model(head_path)
    with open(classes_path) as f:
        class_names = json.load(f)

    # Warm up the exact code path a real request takes (including the
    # Grad-CAM GradientTape call, not just a plain forward pass) so any
    # first-call tracing quirks happen once here at startup, not on a
    # user's first request.
    grad_model = build_grad_model(base_model)
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    compute_gradcam(dummy, grad_model, head_model)

    _state["base_model"] = base_model
    _state["head_model"] = head_model
    _state["grad_model"] = grad_model
    _state["class_names"] = class_names
    _state["preprocess_input"] = preprocess_input
    _state["loaded"] = True
    app.logger.info("VGG16 model loaded successfully: classes=%s", class_names)


def _load_efficientnet_model():
    from efficientnet_infer import build_gradcam, load_model, predict_with_gradcam

    checkpoint_path = os.path.join(EFFICIENTNET_DIR, "brain_tumor_efficientnet_best.pth")
    classes_path = os.path.join(EFFICIENTNET_DIR, "class_indices.json")

    if not (os.path.exists(checkpoint_path) and os.path.exists(classes_path)):
        _state["load_error"] = (
            "Trained EfficientNet checkpoint not found in model/efficientnet/. "
            "Run 'new model/main.py' first."
        )
        return

    with open(classes_path) as f:
        class_names = json.load(f)

    model = load_model(checkpoint_path, len(class_names))
    gradcam = build_gradcam(model)

    # Warm up the exact code path a real request takes (including the
    # Grad-CAM backward pass), so any first-call overhead happens once here
    # at startup, not on a user's first request.
    dummy = Image.new("RGB", (224, 224))
    predict_with_gradcam(dummy, dummy, gradcam, class_names)

    _state["torch_model"] = model
    _state["gradcam"] = gradcam
    _state["class_names"] = class_names
    _state["loaded"] = True
    app.logger.info("EfficientNet model loaded successfully: classes=%s", class_names)


def _load_models():
    if _state["loaded"]:
        return
    _state["load_error"] = None
    try:
        if _state["backend"] == "efficientnet":
            _load_efficientnet_model()
        elif _state["backend"] == "vgg16":
            _load_vgg16_model()
        else:
            _state["load_error"] = (
                f"Unknown MODEL_BACKEND '{_state['backend']}' (expected 'efficientnet' or 'vgg16')."
            )
    except Exception as e:  # noqa: BLE001 - surfaced to /api/health and /api/predict
        _state["load_error"] = str(e)
        app.logger.exception("Failed to load model")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_data_uri(img_array_uint8):
    img = Image.fromarray(img_array_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def resize_for_display(img: Image.Image):
    w, h = img.size
    scale = DISPLAY_MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


@app.route("/api/health")
def health():
    _load_models()
    return jsonify({
        "status": "ok",
        "model_loaded": _state["loaded"],
        "error": _state["load_error"],
        "backend": _state["backend"],
    })


@app.route("/api/model-info")
def model_info():
    _load_models()
    if _state["load_error"]:
        return jsonify({"error": _state["load_error"]}), 503

    report_dir = EFFICIENTNET_DIR if _state["backend"] == "efficientnet" else MODEL_DIR
    metrics = None
    report_path = os.path.join(report_dir, "evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            metrics = json.load(f)

    return jsonify({
        "classes": _state["class_names"],
        "metrics": metrics,
        "backend": _state["backend"],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    _load_models()
    if _state["load_error"]:
        return jsonify({"error": _state["load_error"]}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided (expected form field 'image')."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Please upload a PNG, JPG or BMP image."}), 400

    try:
        raw = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    display_img = resize_for_display(raw)
    display_arr = np.array(display_img).astype(np.uint8)
    class_names = _state["class_names"]

    if _state["backend"] == "efficientnet":
        from efficientnet_infer import overlay_heatmap_on_image, predict_with_gradcam
        heatmap, pred_idx, probs = predict_with_gradcam(raw, display_img, _state["gradcam"], class_names)
    else:
        from gradcam import compute_gradcam, overlay_heatmap_on_image
        model_img = raw.resize(IMG_SIZE, Image.LANCZOS)
        model_arr = np.array(model_img).astype(np.float32)
        batch = _state["preprocess_input"](np.expand_dims(model_arr, axis=0))
        heatmap, pred_idx, probs = compute_gradcam(batch, _state["grad_model"], _state["head_model"])

    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    overlay_arr = overlay_heatmap_on_image(heatmap, display_arr)

    return jsonify({
        "prediction": pred_class,
        "tumor_detected": pred_class != "notumor",
        "confidence": confidence,
        "probabilities": {cls: float(p) for cls, p in zip(class_names, probs)},
        "original_image": image_to_data_uri(display_arr),
        "heatmap_overlay": image_to_data_uri(overlay_arr),
    })


if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
