"""Grad-CAM utilities shared by the training script (for report figures)
and the Flask backend (for live per-prediction heatmaps).

Kept as two separate models (conv feature extractor + dense classifier
head) rather than one nested model, since Grad-CAM needs gradients of the
prediction with respect to an *intermediate* conv layer, and Keras does
not expose intermediate tensors of a model that has been nested as a
single opaque layer inside another model.
"""
import numpy as np
import tensorflow as tf
import cv2

LAST_CONV_LAYER = "block5_conv3"


def build_grad_model(base_model, last_conv_layer_name=LAST_CONV_LAYER):
    """Build once and reuse across requests/calls.

    Rebuilding a functional tf.keras.Model from an existing model's layers
    on every call is not free - Keras has to re-walk and re-register the
    layer graph each time, which measurably slows down (and, over many
    repeated calls in a long-lived process such as the Flask server,
    progressively degrades) if done per-request instead of once at
    startup.
    """
    return tf.keras.Model(
        base_model.input,
        [base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )


def compute_gradcam(preprocessed_batch, grad_model, head_model, class_index=None):
    """preprocessed_batch: (1, 224, 224, 3) float32, already run through
    keras.applications.vgg16.preprocess_input.
    grad_model: pre-built via build_grad_model(base_model) - build it once
    and pass the same instance on every call.

    Returns (heatmap in [0,1] at conv resolution, predicted class index,
    full probability vector).
    """
    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(preprocessed_batch)
        preds = head_model(base_output)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / max_val if max_val > 0 else heatmap
    return heatmap.numpy(), class_index, preds.numpy()[0]


def overlay_heatmap_on_image(heatmap, original_rgb_uint8, alpha=0.4):
    """original_rgb_uint8: HxWx3 uint8 RGB image at display resolution."""
    h, w = original_rgb_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    overlay = (
        colored_rgb.astype(np.float32) * alpha
        + original_rgb_uint8.astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)
    return overlay
