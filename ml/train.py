"""Brain tumour MRI classifier training pipeline.

Implements the methodology from the project report (Chapter 5/6): VGG16
transfer learning with frozen-base bottleneck training followed by
fine-tuning of the last conv block, on the 4-class Kaggle brain MRI
dataset (glioma, meningioma, notumor, pituitary) already present under
dataset/Training and dataset/Testing.

Two-stage strategy (chosen for CPU-only training speed):
  Stage 1: freeze the VGG16 conv base, run every image through it once to
           get "bottleneck" features, then train just the small dense
           classifier head on those cached features (seconds per epoch
           instead of minutes, since the expensive conv forward pass only
           happens once).
  Stage 2: unfreeze the last conv block (block5) and fine-tune the whole
           thing end-to-end at a low learning rate with augmentation, as
           described in report section 6.3 ("top layers are retrained").

The conv base and the dense head are kept as two separate saved models
(model/vgg16_base.keras, model/classifier_head.keras) rather than one
merged model, because Grad-CAM (see gradcam.py) needs direct access to an
intermediate conv layer's output, which is not reachable through Keras's
graph once a submodel is nested as a single opaque layer.
"""
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "Training")
TEST_DIR = os.path.join(DATASET_DIR, "Testing")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
AUTOTUNE = tf.data.AUTOTUNE

HEAD_EPOCHS = 30
HEAD_PATIENCE = 6
FT_EPOCHS = 6
FT_PATIENCE = 2
FT_LR = 1e-5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_raw_dataset(directory, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )


def build_head_model(feature_shape, num_classes):
    inputs = layers.Input(shape=feature_shape, name="bottleneck_features")
    x = layers.Flatten(name="flatten")(inputs)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="classifier_head")


def extract_features(dataset, base_model, name):
    log(f"Extracting bottleneck features for {name} set...")
    features_list, labels_list = [], []
    n_batches = 0
    for x_batch, y_batch in dataset:
        x_pre = preprocess_input(x_batch)
        feats = base_model(x_pre, training=False).numpy()
        features_list.append(feats)
        labels_list.append(y_batch.numpy())
        n_batches += 1
        if n_batches % 20 == 0:
            log(f"  {name}: {n_batches} batches processed")
    log(f"  {name}: done ({n_batches} batches)")
    return np.concatenate(features_list), np.concatenate(labels_list)


def plot_training_curves(history_stage1, history_stage2, out_path):
    epochs1 = len(history_stage1["accuracy"])
    epochs2 = len(history_stage2["accuracy"])

    acc = history_stage1["accuracy"] + history_stage2["accuracy"]
    val_acc = history_stage1["val_accuracy"] + history_stage2["val_accuracy"]
    loss = history_stage1["loss"] + history_stage2["loss"]
    val_loss = history_stage1["val_loss"] + history_stage2["val_loss"]
    x = list(range(1, len(acc) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, series_train, series_val, title in (
        (axes[0], acc, val_acc, "Accuracy"),
        (axes[1], loss, val_loss, "Loss"),
    ):
        ax.plot(x, series_train, label="Training", color="#2563eb")
        ax.plot(x, series_val, label="Validation", color="#f97316")
        ax.axvline(epochs1 + 0.5, color="gray", linestyle="--", linewidth=1,
                   label="Fine-tuning starts" if title == "Accuracy" else None)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Training History: Head Training + Fine-Tuning")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved training curves to {out_path}")


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved confusion matrix to {out_path}")


def save_gradcam_examples(base_model, head_model, out_path):
    from gradcam import build_grad_model, compute_gradcam, overlay_heatmap_on_image
    import random

    grad_model = build_grad_model(base_model)
    random.seed(7)
    fig, axes = plt.subplots(2, len(CLASS_NAMES), figsize=(4 * len(CLASS_NAMES), 8))
    for col, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(TEST_DIR, cls)
        fname = random.choice(os.listdir(cls_dir))
        img_path = os.path.join(cls_dir, fname)
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        img_arr = tf.keras.utils.img_to_array(img).astype(np.uint8)
        batch = preprocess_input(np.expand_dims(img_arr.astype(np.float32), 0))
        heatmap, pred_idx, probs = compute_gradcam(batch, grad_model, head_model)
        overlay = overlay_heatmap_on_image(heatmap, img_arr)

        axes[0, col].imshow(img_arr)
        axes[0, col].set_title(f"True: {cls}")
        axes[0, col].axis("off")
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"Pred: {CLASS_NAMES[pred_idx]} ({probs[pred_idx]*100:.1f}%)")
        axes[1, col].axis("off")
    fig.suptitle("Grad-CAM: Original MRI (top) vs. Model Attention Heatmap (bottom)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved Grad-CAM example grid to {out_path}")


def main():
    start_time = time.time()
    log(f"TensorFlow {tf.__version__}, classes: {CLASS_NAMES}")

    train_ds_raw = make_raw_dataset(TRAIN_DIR, shuffle=True)
    val_ds_raw = make_raw_dataset(TEST_DIR, shuffle=False)

    # ---------------- Stage 1: bottleneck feature extraction ----------------
    log("Loading ImageNet-pretrained VGG16 conv base...")
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    train_features, train_labels = extract_features(train_ds_raw, base_model, "train")
    val_features, val_labels = extract_features(val_ds_raw, base_model, "validation")
    log(f"Bottleneck feature shapes: train={train_features.shape}, val={val_features.shape}")

    head_model = build_head_model(train_features.shape[1:], len(CLASS_NAMES))
    head_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    head_model.summary(print_fn=log)

    es_head = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=HEAD_PATIENCE, restore_best_weights=True
    )
    log("Training classifier head on cached bottleneck features (Stage 1)...")
    hist1 = head_model.fit(
        train_features, train_labels,
        validation_data=(val_features, val_labels),
        epochs=HEAD_EPOCHS,
        batch_size=32,
        callbacks=[es_head],
        verbose=2,
    )
    log(f"Stage 1 complete in {time.time() - start_time:.1f}s")

    # ---------------- Stage 2: fine-tune last conv block ----------------
    log("Unfreezing block5 of VGG16 for fine-tuning (Stage 2)...")
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = layer.name.startswith("block5")

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="augmentation")

    def prep_train(x, y):
        x = data_augmentation(x, training=True)
        return preprocess_input(x), y

    def prep_val(x, y):
        return preprocess_input(x), y

    train_ds_ft = train_ds_raw.map(prep_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds_ft = val_ds_raw.map(prep_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(FT_LR)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    train_loss_metric = tf.keras.metrics.Mean()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_loss_metric = tf.keras.metrics.Mean()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            features = base_model(x, training=True)
            preds = head_model(features, training=True)
            loss = loss_fn(y, preds)
        trainable_vars = base_model.trainable_variables + head_model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        train_acc_metric.update_state(y, preds)
        train_loss_metric.update_state(loss)

    @tf.function
    def val_step(x, y):
        features = base_model(x, training=False)
        preds = head_model(features, training=False)
        loss = loss_fn(y, preds)
        val_acc_metric.update_state(y, preds)
        val_loss_metric.update_state(loss)

    best_val_acc = -1.0
    wait = 0
    best_base_path = os.path.join(MODEL_DIR, "_best_base.weights.h5")
    best_head_path = os.path.join(MODEL_DIR, "_best_head.weights.h5")
    hist2 = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(FT_EPOCHS):
        epoch_start = time.time()
        train_acc_metric.reset_state()
        train_loss_metric.reset_state()
        for x_batch, y_batch in train_ds_ft:
            train_step(x_batch, y_batch)

        val_acc_metric.reset_state()
        val_loss_metric.reset_state()
        for x_batch, y_batch in val_ds_ft:
            val_step(x_batch, y_batch)

        tr_acc = float(train_acc_metric.result())
        tr_loss = float(train_loss_metric.result())
        v_acc = float(val_acc_metric.result())
        v_loss = float(val_loss_metric.result())
        hist2["accuracy"].append(tr_acc)
        hist2["loss"].append(tr_loss)
        hist2["val_accuracy"].append(v_acc)
        hist2["val_loss"].append(v_loss)

        log(f"Fine-tune epoch {epoch+1}/{FT_EPOCHS} - loss {tr_loss:.4f} - acc {tr_acc:.4f} "
            f"- val_loss {v_loss:.4f} - val_acc {v_acc:.4f} - {time.time()-epoch_start:.1f}s")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            wait = 0
            base_model.save_weights(best_base_path)
            head_model.save_weights(best_head_path)
        else:
            wait += 1
            if wait >= FT_PATIENCE:
                log("Early stopping fine-tuning stage.")
                break

    if os.path.exists(best_base_path):
        base_model.load_weights(best_base_path)
        head_model.load_weights(best_head_path)
        os.remove(best_base_path)
        os.remove(best_head_path)

    # ---------------- Save final models ----------------
    base_model.save(os.path.join(MODEL_DIR, "vgg16_base.keras"))
    head_model.save(os.path.join(MODEL_DIR, "classifier_head.keras"))
    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)
    log("Saved vgg16_base.keras, classifier_head.keras, class_indices.json")

    # ---------------- Evaluation ----------------
    log("Running final evaluation on test set...")
    y_true, y_pred_probs = [], []
    for x_batch, y_batch in val_ds_ft:
        feats = base_model(x_batch, training=False)
        probs = head_model(feats, training=False).numpy()
        y_pred_probs.append(probs)
        y_true.append(y_batch.numpy())
    y_true = np.concatenate(y_true)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_true_idx, y_pred_idx, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    with open(os.path.join(MODEL_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(json.dumps(report["accuracy"] if "accuracy" in report else {}, indent=2))
    log(f"Overall test accuracy: {report['accuracy']*100:.2f}%")

    cm = confusion_matrix(y_true_idx, y_pred_idx)
    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(MODEL_DIR, "confusion_matrix.png"))

    hist1_dict = {
        "accuracy": hist1.history["accuracy"],
        "val_accuracy": hist1.history["val_accuracy"],
        "loss": hist1.history["loss"],
        "val_loss": hist1.history["val_loss"],
    }
    plot_training_curves(hist1_dict, hist2, os.path.join(MODEL_DIR, "training_curves.png"))

    try:
        save_gradcam_examples(base_model, head_model, os.path.join(MODEL_DIR, "gradcam_examples.png"))
    except Exception as e:
        log(f"Grad-CAM example generation failed (non-fatal): {e}")

    total_time = time.time() - start_time
    log(f"ALL DONE in {total_time/60:.1f} minutes. Final test accuracy: {report['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
