"""Integration test: drives the *live* Flask API (not the training script's
in-process evaluation) across the entire dataset/Testing/ set, one HTTP
request per image, exactly as the frontend would call it.

This exists to catch integration bugs that a pure model-level evaluation
can't see: wrong preprocessing in the API path, wrong class-index mapping,
JPEG re-encoding artifacts, off-by-one errors in the two-model (base +
head) split, etc. Compare its accuracy against model/evaluation_report.json
(the training script's own held-out evaluation) - they should closely agree.

Usage:  venv\\Scripts\\python.exe ml\\evaluate_api.py [--limit N] [--api http://localhost:5000]
"""
import argparse
import json
import os
import time

import requests
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "Testing")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def collect_test_files(limit_per_class=None):
    files = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(TEST_DIR, cls)
        names = sorted(os.listdir(cls_dir))
        if limit_per_class:
            names = names[:limit_per_class]
        for name in names:
            files.append((os.path.join(cls_dir, name), cls))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="max images per class (default: use all)")
    parser.add_argument("--api", default="http://localhost:5000")
    args = parser.parse_args()

    health = requests.get(f"{args.api}/api/health", timeout=10).json()
    if not health.get("model_loaded"):
        raise SystemExit(f"Backend model not loaded: {health}")

    # The live server reports which model it's serving; write results next
    # to that backend's own artifacts (model/efficientnet/ or model/) so
    # this always compares against the right offline evaluation_report.json.
    backend = health.get("backend", "vgg16")
    report_dir = os.path.join(MODEL_DIR, "efficientnet") if backend == "efficientnet" else MODEL_DIR
    print(f"Backend under test: {backend}")

    files = collect_test_files(args.limit)
    print(f"Testing {len(files)} images against the live API at {args.api} ...")

    y_true, y_pred, failures = [], [], []
    start = time.time()

    for i, (path, true_cls) in enumerate(files, 1):
        with open(path, "rb") as f:
            resp = requests.post(
                f"{args.api}/api/predict",
                files={"image": (os.path.basename(path), f, "image/jpeg")},
                timeout=60,
            )
        if resp.status_code != 200:
            failures.append((path, resp.status_code, resp.text[:200]))
            continue
        body = resp.json()
        y_true.append(true_cls)
        y_pred.append(body["prediction"])

        if i % 100 == 0 or i == len(files):
            elapsed = time.time() - start
            print(f"  {i}/{len(files)} done ({elapsed:.1f}s, {elapsed/i:.2f}s/img)")

    elapsed = time.time() - start
    print(f"\nFinished {len(y_true)}/{len(files)} requests in {elapsed:.1f}s "
          f"({len(failures)} failed)")

    if failures:
        print("\nFailed requests (first 5):")
        for path, code, text in failures[:5]:
            print(f"  {path}: HTTP {code} - {text}")

    if not y_true:
        raise SystemExit("No successful predictions - aborting.")

    report = classification_report(y_true, y_pred, labels=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

    print("\n=== Live API classification report ===")
    print(classification_report(y_true, y_pred, labels=CLASS_NAMES, zero_division=0))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print("           " + "  ".join(f"{c[:10]:>10}" for c in CLASS_NAMES))
    for cls, row in zip(CLASS_NAMES, cm):
        print(f"{cls:>10} " + "  ".join(f"{v:>10}" for v in row))

    out = {
        "api_accuracy": report["accuracy"],
        "n_images": len(y_true),
        "n_failures": len(failures),
        "elapsed_seconds": elapsed,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    out_path = os.path.join(report_dir, "api_evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    offline_report_path = os.path.join(report_dir, "evaluation_report.json")
    if os.path.exists(offline_report_path):
        with open(offline_report_path) as f:
            offline = json.load(f)
        print(f"\nLive API accuracy:   {report['accuracy']*100:.2f}%")
        print(f"Offline eval accuracy: {offline['accuracy']*100:.2f}% (from ml/train.py's own held-out evaluation)")


if __name__ == "__main__":
    main()
