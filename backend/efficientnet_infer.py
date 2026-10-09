"""PyTorch EfficientNet-B0 inference + Grad-CAM for the Flask backend.

Mirrors the architecture, checkpoint format and preprocessing used by
"new model/main.py" (training script) so that predictions made here match
what the model was actually trained on: brain-region crop -> resize to
224x224 -> ImageNet normalization.

Kept separate from "new model/gradcam.py" (used for offline report figures)
to avoid a module-name collision with ml/gradcam.py (the TensorFlow Grad-CAM
helper the Flask app also imports as `gradcam` for the VGG16 backend) and to
return a Grad-CAM heatmap already mapped back onto the *uncropped* display
image, which the report-figure script has no need for.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 224
_TO_TENSOR = transforms.ToTensor()
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def crop_brain_region_bbox(img, threshold=10, padding=4):
    """Bounding box (x0, y0, x1, y1) of the non-black brain tissue region.

    Returns a box instead of the cropped image (unlike the CropBrainRegion
    used at training time) so a Grad-CAM heatmap computed on the crop can be
    pasted back at the right location in the full, uncropped display image.
    """
    gray = np.array(img.convert("L"))
    mask = gray > threshold
    if mask.sum() == 0:
        return 0, 0, img.width, img.height
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    h, w = gray.shape
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(h, y1 + padding)
    x1 = min(w, x1 + padding)
    return int(x0), int(y0), int(x1), int(y1)


def build_model(num_classes):
    """Same architecture as the classifier head trained in new model/main.py."""
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, num_classes),
    )
    return model


def load_model(checkpoint_path, num_classes):
    model = build_model(num_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


class GradCAM:
    """Standard Grad-CAM hooked onto a target conv layer, matching
    "new model/gradcam.py"'s GradCAM class.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(output[0]))
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam.cpu().numpy(), class_idx, probs

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


def build_gradcam(model):
    """Target layer: model.features[-1], the final 1x1 "head" conv of
    EfficientNet-B0 - highest-level semantic features while still retaining
    a spatial map to visualize. Matches new model/gradcam.py.
    """
    return GradCAM(model, model.features[-1])


def _resize_cam(cam, size):
    """cam: 2D numpy array in [0, 1]. size: (w, h). Returns float32 array."""
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.array(cam_img).astype(np.float32) / 255.0


def predict_with_gradcam(raw_img, display_img, gradcam, class_names):
    """raw_img: PIL RGB image at its original, native resolution.
    display_img: raw_img resized to a display-friendly resolution (may be
    identical to raw_img if it was already small enough).

    The brain-region crop bounding box MUST be computed on raw_img, not on
    a possibly-downscaled display_img: training (new model/main.py's
    CropBrainRegion) always runs on the native-resolution file, and its
    fixed 4px padding means cropping a downscaled copy instead shifts the
    box relative to what the model was trained on - a real train/serve
    skew for any image whose longer side exceeds DISPLAY_MAX_SIDE.

    Returns (heatmap in [0, 1] sized to display_img, pred_idx, probs) - the
    heatmap is zero everywhere outside the detected brain region bounding
    box (mapped into display_img's coordinate system), since that's the
    only area the model actually looked at.
    """
    x0, y0, x1, y1 = crop_brain_region_bbox(raw_img)
    cropped = raw_img.crop((x0, y0, x1, y1))
    model_img = cropped.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    tensor = _NORMALIZE(_TO_TENSOR(model_img)).unsqueeze(0)

    cam, pred_idx, probs = gradcam.generate(tensor)

    scale_x = display_img.width / raw_img.width
    scale_y = display_img.height / raw_img.height
    dx0, dy0 = int(x0 * scale_x), int(y0 * scale_y)
    dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
    dx1, dy1 = max(dx1, dx0 + 1), max(dy1, dy0 + 1)

    cam_resized = _resize_cam(cam, (dx1 - dx0, dy1 - dy0))
    full_heatmap = np.zeros((display_img.height, display_img.width), dtype=np.float32)
    full_heatmap[dy0:dy1, dx0:dx1] = cam_resized

    return full_heatmap, pred_idx, probs


def overlay_heatmap_on_image(heatmap, original_rgb_uint8, alpha=0.4):
    """original_rgb_uint8: HxWx3 uint8 RGB image at display resolution.

    Identical in behavior to ml/gradcam.py's version of this function (pure
    numpy + cv2, no framework dependency) - duplicated here rather than
    imported so this module never pulls in TensorFlow just to serve the
    EfficientNet backend.
    """
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
