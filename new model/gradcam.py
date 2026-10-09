import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

# Same project layout as main.py: this script lives in "<repo>/new model/".
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
TEST_DIR = os.path.join(REPO_DIR, "dataset", "Testing")
OUT_DIR = os.path.join(REPO_DIR, "model", "efficientnet")
os.makedirs(OUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUT_DIR, "brain_tumor_efficientnet_best.pth")


# Must match the CropBrainRegion used in training exactly, so Grad-CAM is
# generated on the same input distribution the model actually learned from.
class CropBrainRegion:
    def __init__(self, threshold=10, padding=4):
        self.threshold = threshold
        self.padding = padding

    def __call__(self, img):
        gray = np.array(img.convert('L'))
        mask = gray > self.threshold
        if mask.sum() == 0:
            return img
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        h, w = gray.shape
        y0 = max(0, y0 - self.padding)
        x0 = max(0, x0 - self.padding)
        y1 = min(h, y1 + self.padding)
        x1 = min(w, x1 + self.padding)
        return img.crop((x0, y0, x1, y1))


class GradCAM:
    """
    Standard Grad-CAM: hooks the target conv layer's activations and
    gradients, then weights each activation channel by the average
    gradient flowing into it for the class of interest.
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

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))  # (C,) -- global-avg-pooled gradients

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        probs = torch.softmax(output, dim=1)
        return cam.cpu().numpy(), probs[0, class_idx].item(), output

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


def load_image_for_gradcam(path, crop_fn, img_size=224):
    """Returns (normalized model input tensor, un-normalized display image)."""
    img = Image.open(path).convert('RGB')
    img = crop_fn(img)
    img = img.resize((img_size, img_size))
    display_img = np.array(img).astype(np.float32) / 255.0

    tensor = transforms.ToTensor()(img)
    tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(tensor)
    return tensor.unsqueeze(0), display_img


def overlay_heatmap(display_img, cam, alpha=0.45):
    h, w = display_img.shape[:2]
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0
    heatmap = cm.jet(cam_resized)[..., :3]
    overlay = (1 - alpha) * display_img + alpha * heatmap
    return np.clip(overlay, 0, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    random.seed(42)

    TEST_ROOT = TEST_DIR
    IMG_SIZE = 224
    SAMPLES_PER_CLASS = 3

    # -------------------------------------------------------------
    # 1. Load class names + file list (no DataLoader needed here --
    #    we load images one at a time to pair each with its Grad-CAM)
    # -------------------------------------------------------------
    test_data = datasets.ImageFolder(root=TEST_ROOT)
    class_names = test_data.classes
    print(f"Classes: {class_names}")

    # -------------------------------------------------------------
    # 2. Rebuild model architecture, load trained weights
    # -------------------------------------------------------------
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, len(class_names))
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Target layer: the final conv block in EfficientNet-B0's feature
    # extractor (features[8], the 1x1 "head" conv before pooling) --
    # highest-level semantic features while still retaining a 7x7 spatial
    # map to visualize.
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    crop_fn = CropBrainRegion()

    # -------------------------------------------------------------
    # 3. Correctly-classified examples per class
    # -------------------------------------------------------------
    samples_by_class = {i: [] for i in range(len(class_names))}
    for path, label in test_data.samples:
        samples_by_class[label].append(path)
    for label in samples_by_class:
        random.shuffle(samples_by_class[label])

    correct_examples = []  # (path, true_idx, pred_idx, cam, display_img, conf)
    misclassified_examples = []

    for class_idx, class_name in enumerate(class_names):
        found_correct = 0
        for path in samples_by_class[class_idx]:
            input_tensor, display_img = load_image_for_gradcam(path, crop_fn, IMG_SIZE)
            input_tensor = input_tensor.to(device)

            cam, conf, output = gradcam.generate(input_tensor, class_idx)
            pred_idx = output.argmax(dim=1).item()

            if pred_idx == class_idx and found_correct < SAMPLES_PER_CLASS:
                correct_examples.append((path, class_idx, pred_idx, cam, display_img, conf))
                found_correct += 1
            elif pred_idx != class_idx and len(misclassified_examples) < 8:
                # Re-generate CAM for the class it actually PREDICTED --
                # shows what feature convinced the model, which is more
                # informative than the CAM for the true label it missed.
                cam_pred, conf_pred, _ = gradcam.generate(input_tensor, pred_idx)
                misclassified_examples.append(
                    (path, class_idx, pred_idx, cam_pred, display_img, conf_pred)
                )

            if found_correct >= SAMPLES_PER_CLASS:
                break

    gradcam.remove_hooks()

    # -------------------------------------------------------------
    # 4. Plot: correctly classified examples, grouped by class
    # -------------------------------------------------------------
    n_rows = len(correct_examples)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for row, (path, true_idx, pred_idx, cam, display_img, conf) in enumerate(correct_examples):
        overlay = overlay_heatmap(display_img, cam)

        axes[row, 0].imshow(display_img)
        axes[row, 0].set_title(f"{class_names[true_idx]}\n(original)", fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"Grad-CAM  (conf={conf:.2f})", fontsize=10)
        axes[row, 1].axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'gradcam_correct_predictions.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved '{out_path}'")

    # -------------------------------------------------------------
    # 5. Plot: misclassified examples (if any), showing what the
    #    model saw that made it pick the WRONG class
    # -------------------------------------------------------------
    if misclassified_examples:
        n_rows = len(misclassified_examples)
        fig, axes = plt.subplots(n_rows, 2, figsize=(6, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, 2)

        for row, (path, true_idx, pred_idx, cam, display_img, conf) in enumerate(misclassified_examples):
            overlay = overlay_heatmap(display_img, cam)

            axes[row, 0].imshow(display_img)
            axes[row, 0].set_title(f"True: {class_names[true_idx]}\n(original)", fontsize=10)
            axes[row, 0].axis('off')

            axes[row, 1].imshow(overlay)
            axes[row, 1].set_title(
                f"Predicted: {class_names[pred_idx]} (conf={conf:.2f})", fontsize=10
            )
            axes[row, 1].axis('off')

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, 'gradcam_misclassified.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved '{out_path}'")
    else:
        print("No misclassified examples found to visualize.")

    print("\nDone. Grad-CAM heatmaps highlight the regions (in red/yellow) "
          "that most influenced each prediction.")


if __name__ == "__main__":
    main()
