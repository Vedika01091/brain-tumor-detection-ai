"""Flask API for the Brain Tumor Detection system.

Serves the fine-tuned PyTorch EfficientNet-B0 model (model/efficientnet_v2/) and returns, for each uploaded
MRI slice, the predicted class, the probability of every class and a Grad-CAM heatmap overlay.

Routes:
  GET  /api/health        - liveness + whether the trained model is loaded
  GET  /api/model-info     - class list + saved evaluation metrics
  POST /api/predict        - multipart image upload -> prediction + Grad-CAM

Run with:  venv/bin/python backend/app.py   (Windows: venv\\Scripts\\python.exe backend\\app.py)

Environment variables: EFFICIENTNET_CHECKPOINT (use another .pth file or extracted .pth folder),
BACKEND_HOST / BACKEND_PORT (default 127.0.0.1 / 5000), FLASK_DEBUG=1 (development only).
"""
import base64
import io
import json
import os
import threading

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "efficientnet_v2")
DEFAULT_CHECKPOINT = os.path.join(MODEL_DIR, "brain_tumor_efficientnet_v2.pth")
CLASSES_PATH = os.path.join(MODEL_DIR, "class_indices.json")
REPORT_PATH = os.path.join(MODEL_DIR, "evaluation_report.json")

DISPLAY_MAX_SIDE = 512
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
CORS(app)

_state = {
    "loaded": False,
    "load_error": None,
    "class_names": None,
    "torch_model": None,
    "gradcam": None,
}

# Flask's dev server handles requests on several threads, but the model and its Grad-CAM hooks keep
# per-call state on shared objects. Running two inferences at once corrupted that state and crashed the
# process, so model loading and inference are serialised with this lock.
_infer_lock = threading.RLock()


def _load_models():
    with _infer_lock:
        _load_models_locked()


def _load_models_locked():
    if _state["loaded"]:
        return
    _state["load_error"] = None
    try:
        from efficientnet_infer import build_gradcam, load_model, predict_with_gradcam

        checkpoint_path = os.environ.get("EFFICIENTNET_CHECKPOINT") or DEFAULT_CHECKPOINT
        if not (os.path.exists(checkpoint_path) and os.path.exists(CLASSES_PATH)):
            _state["load_error"] = f"Trained model not found: {checkpoint_path}"
            return

        with open(CLASSES_PATH) as f:
            class_names = json.load(f)

        model = load_model(checkpoint_path, len(class_names))
        gradcam = build_gradcam(model)

        # Warm up the exact code path a real request takes (including the Grad-CAM backward pass),
        # so first-call overhead happens once here and not on a user's first request.
        dummy = Image.new("RGB", (224, 224))
        predict_with_gradcam(dummy, dummy, gradcam, class_names)

        _state["torch_model"] = model
        _state["gradcam"] = gradcam
        _state["class_names"] = class_names
        _state["loaded"] = True
        app.logger.info("EfficientNet model loaded successfully: classes=%s", class_names)
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


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File is too large (maximum 16 MB)."}), 413


@app.route("/api/health")
def health():
    _load_models()
    return jsonify({
        "status": "ok",
        "model_loaded": _state["loaded"],
        "error": _state["load_error"],
        "backend": "efficientnet",
    })


@app.route("/api/model-info")
def model_info():
    _load_models()
    if _state["load_error"]:
        return jsonify({"error": _state["load_error"]}), 503

    metrics = None
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            metrics = json.load(f)

    return jsonify({
        "classes": _state["class_names"],
        "metrics": metrics,
        "backend": "efficientnet",
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

    from efficientnet_infer import overlay_heatmap_on_image, predict_with_gradcam
    with _infer_lock:
        heatmap, pred_idx, probs = predict_with_gradcam(raw, display_img, _state["gradcam"], class_names)

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
    # Debug mode exposes the Werkzeug interactive debugger, so it is opt-in
    # (FLASK_DEBUG=1) and the server only listens on localhost unless
    # BACKEND_HOST says otherwise (e.g. 0.0.0.0 inside a container).
    app.run(
        host=os.environ.get("BACKEND_HOST", "127.0.0.1"),
        port=int(os.environ.get("BACKEND_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
        use_reloader=False,
    )
