import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from PIL import Image

# Same project layout as main.py: this script lives in "<repo>/new model/".
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
TEST_DIR = os.path.join(REPO_DIR, "dataset", "Testing")
OUT_DIR = os.path.join(REPO_DIR, "model", "efficientnet")
os.makedirs(OUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUT_DIR, "brain_tumor_efficientnet_best.pth")


# Must match the CropBrainRegion used in brain_tumor_training_v3.py exactly,
# or evaluation will see a different input distribution than the model was
# trained on.
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------------
    # 1. Test data (must match the eval transforms used in training)
    # ---------------------------------------------------------------
    eval_transforms = transforms.Compose([
        CropBrainRegion(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_data = datasets.ImageFolder(root=TEST_DIR, transform=eval_transforms)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
    class_names = test_data.classes
    print(f"Classes: {class_names}")
    print(f"Test set size: {len(test_data)}")

    # ---------------------------------------------------------------
    # 2. Rebuild model architecture, then load trained weights
    # ---------------------------------------------------------------
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, len(class_names))
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # ---------------------------------------------------------------
    # 3. Run inference, collect predictions + misclassified file paths
    # ---------------------------------------------------------------
    all_preds = []
    all_labels = []
    misclassified = []  # (filepath, true_label, pred_label, confidence)

    sample_idx = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_dev = inputs.to(device)
            outputs = model(inputs_dev)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            preds_cpu = preds.cpu()
            confs_cpu = confs.cpu()

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds_cpu[i].item()
                if true_label != pred_label:
                    filepath = test_data.samples[sample_idx][0]
                    misclassified.append((filepath, true_label, pred_label, confs_cpu[i].item()))
                sample_idx += 1

            all_preds.append(preds_cpu)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ---------------------------------------------------------------
    # 4. Per-class metrics
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    with open(os.path.join(OUT_DIR, 'classification_report_v3.txt'), 'w') as f:
        f.write(report)

    # ---------------------------------------------------------------
    # 5. Confusion matrix — raw counts + normalized heatmap
    # ---------------------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (rows=true, cols=predicted)")
    print("=" * 60)
    header = "".join(f"{name[:12]:>14}" for name in class_names)
    print(f"{'':>14}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>14}" for v in row)
        print(f"{class_names[i][:12]:>14}{row_str}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix (row-normalized)')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                    ha='center', va='center', color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, label='Proportion of true class')
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, 'confusion_matrix_v3.png')
    plt.savefig(cm_path, dpi=150)
    print(f"\nSaved heatmap to '{cm_path}'")

    # ---------------------------------------------------------------
    # 6. Which specific class-pairs are being confused most?
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TOP CONFUSION PAIRS (excluding correct predictions)")
    print("=" * 60)
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    for count, true_cls, pred_cls in pairs[:10]:
        print(f"  True={true_cls:<12} -> Predicted={pred_cls:<12}  count={count}")

    # ---------------------------------------------------------------
    # 7. Lowest-confidence misclassifications (worth manually inspecting)
    # ---------------------------------------------------------------
    misclassified.sort(key=lambda x: -x[3])  # highest-confidence wrong predictions first
    print("\n" + "=" * 60)
    print(f"MISCLASSIFIED EXAMPLES: {len(misclassified)} total")
    print("Most confident WRONG predictions (model was sure, and wrong):")
    print("=" * 60)
    for filepath, true_idx, pred_idx, conf in misclassified[:15]:
        print(f"  conf={conf:.3f}  true={class_names[true_idx]:<12} "
              f"pred={class_names[pred_idx]:<12}  {filepath}")

    misclassified_path = os.path.join(OUT_DIR, 'misclassified_examples_v3.txt')
    with open(misclassified_path, 'w') as f:
        f.write("confidence,true_label,predicted_label,filepath\n")
        for filepath, true_idx, pred_idx, conf in misclassified:
            f.write(f"{conf:.4f},{class_names[true_idx]},{class_names[pred_idx]},{filepath}\n")
    print(f"\nFull list saved to '{misclassified_path}'")


if __name__ == "__main__":
    main()
