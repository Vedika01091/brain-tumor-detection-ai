import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import copy
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project layout: this script lives in "<repo>/new model/", the dataset in
# "<repo>/dataset/{Training,Testing}", and trained artifacts are written to
# "<repo>/model/efficientnet/" so they sit alongside (but never overwrite)
# the existing VGG16 artifacts in "<repo>/model/".
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
TRAIN_DIR = os.path.join(REPO_DIR, "dataset", "Training")
TEST_DIR = os.path.join(REPO_DIR, "dataset", "Testing")
OUT_DIR = os.path.join(REPO_DIR, "model", "efficientnet")
os.makedirs(OUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUT_DIR, "brain_tumor_efficientnet_best.pth")


# ---------------------------------------------------------------------
# Brain-region crop: MRI scans usually have a black background border
# around the actual brain tissue. Cropping it out before resizing means
# more of the 224x224 input is spent on the tissue itself, instead of
# background -- this matters most for glioma, which has subtle diffuse
# signal that's easy to lose when it's squeezed into a smaller effective
# area by wasted black space.
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Focal loss: down-weights easy, already-confident examples and focuses
# training signal on hard/misclassified ones. Directly targets the
# glioma-recall problem, where the model is "too cautious" -- plain
# cross-entropy lets it stay comfortable predicting the safer classes.
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        return (focal_term * ce_loss).mean()


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    IMG_SIZE = 224  # bump to 260 if your GPU has the VRAM -- higher res
                     # tends to help with subtle glioma signal specifically

    # ---------------------------------------------------------------------
    # 1. Data augmentation (crop happens BEFORE resize in every pipeline)
    # ---------------------------------------------------------------------
    train_transforms = transforms.Compose([
        CropBrainRegion(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        CropBrainRegion(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TTA transforms: a few light augmented views averaged at inference time
    tta_transforms = [
        eval_transforms,
        transforms.Compose([
            CropBrainRegion(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            CropBrainRegion(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation((10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]

    # ---------------------------------------------------------------------
    # 2. IMPORTANT: proper train/val/test split
    #    Test set is NEVER touched until final evaluation.
    #    Val set (carved out of Training/) drives checkpointing + scheduler.
    # ---------------------------------------------------------------------
    full_train_data = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    test_data = datasets.ImageFolder(root=TEST_DIR, transform=eval_transforms)

    val_frac = 0.15
    val_size = int(len(full_train_data) * val_frac)
    train_size = len(full_train_data) - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_train_data, [train_size, val_size], generator=generator)

    # val subset must use eval transforms, not train-time augmentation
    val_subset.dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=eval_transforms)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

    class_names = full_train_data.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_data)}")

    # ---------------------------------------------------------------------
    # 3. Class weights (handles imbalance if present)
    # ---------------------------------------------------------------------
    targets = np.array(full_train_data.targets)
    class_counts = np.bincount(targets)
    class_weights = torch.tensor(
        (len(targets) / (len(class_counts) * class_counts)), dtype=torch.float32
    ).to(device)
    print(f"Class counts: {class_counts}, weights: {class_weights}")

    # ---------------------------------------------------------------------
    # 4. Model: full fine-tune (unfreeze everything) with low LR.
    #    For a domain as different from ImageNet as MRI scans, full
    #    fine-tuning at a low LR usually beats freezing early blocks.
    # ---------------------------------------------------------------------
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = True

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, len(class_names))
    )
    model = model.to(device)

    # ---------------------------------------------------------------------
    # 5. Loss, optimizer, schedule
    #    - label smoothing reduces overconfidence-driven errors
    #    - discriminative LRs: backbone gets a smaller LR than the new head
    #    - OneCycleLR tends to outperform ReduceLROnPlateau for fine-tuning
    # ---------------------------------------------------------------------
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)

    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 3e-5},
        {"params": head_params, "lr": 3e-4},
    ], weight_decay=0.01)

    num_epochs = 20
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[3e-5, 3e-4],
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.1,
    )

    # ---------------------------------------------------------------------
    # 6. Training loop with early stopping on val accuracy
    # ---------------------------------------------------------------------
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 5
    epochs_no_improve = 0
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    print("\n--- Starting Fine-Tuning ---")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        epoch_metrics = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            epoch_metrics[f"{'' if phase == 'train' else 'val_'}loss"] = epoch_loss
            epoch_metrics[f"{'' if phase == 'train' else 'val_'}accuracy"] = float(epoch_acc)

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        history["loss"].append(epoch_metrics["loss"])
        history["accuracy"].append(epoch_metrics["accuracy"])
        history["val_loss"].append(epoch_metrics["val_loss"])
        history["val_accuracy"].append(epoch_metrics["val_accuracy"])

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping: no val improvement for {patience} epochs.")
            break

    print(f"\nTraining Complete! Best Validation Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, CHECKPOINT_PATH)
    print(f"Model saved to '{CHECKPOINT_PATH}'!")

    with open(os.path.join(OUT_DIR, "class_indices.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = list(range(1, len(history["accuracy"]) + 1))
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(x, history[key], label="Training", color="#2563eb")
        ax.plot(x, history[f"val_{key}"], label="Validation", color="#f97316")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("EfficientNet-B0 Fine-Tuning History")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {os.path.join(OUT_DIR, 'training_curves.png')}")

    # ---------------------------------------------------------------------
    # 7. Final, honest evaluation on the untouched test set, with TTA
    #    TTA: average softmax predictions over a few augmented views.
    #    Usually adds ~0.5-1.5% accuracy for free.
    # ---------------------------------------------------------------------
    def evaluate_with_tta(model, test_dataset_root, tta_transforms, device):
        model.eval()
        all_preds = []
        all_labels = []

        # Build one loader per TTA transform, iterate in lockstep
        loaders = [
            DataLoader(
                datasets.ImageFolder(root=test_dataset_root, transform=t),
                batch_size=32, shuffle=False, num_workers=0
            )
            for t in tta_transforms
        ]

        with torch.no_grad():
            for batches in zip(*loaders):
                probs_sum = None
                labels = batches[0][1].to(device)
                for inputs, _ in batches:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    probs_sum = probs if probs_sum is None else probs_sum + probs

                preds = torch.argmax(probs_sum, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = (all_preds == all_labels).float().mean().item()
        return acc


    test_acc_plain = 0.0
    running_corrects = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    test_acc_plain = running_corrects.double() / len(test_data)
    print(f"\nFinal Test Accuracy (no TTA): {test_acc_plain:.4f}")

    test_acc_tta = evaluate_with_tta(model, TEST_DIR, tta_transforms, device)
    print(f"Final Test Accuracy (with TTA): {test_acc_tta:.4f}")

    # ---------------------------------------------------------------------
    # 8. Classification report + confusion matrix, saved in the same shape
    #    as model/evaluation_report.json (the existing VGG16 model) so the
    #    Flask backend and frontend can read either one interchangeably.
    # ---------------------------------------------------------------------
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report["tta_accuracy"] = test_acc_tta
    with open(os.path.join(OUT_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {os.path.join(OUT_DIR, 'evaluation_report.json')}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set) - EfficientNet-B0")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {os.path.join(OUT_DIR, 'confusion_matrix.png')}")

    try:
        from gradcam import GradCAM, load_image_for_gradcam, overlay_heatmap
        import random

        gradcam = GradCAM(model, model.features[-1])
        crop_fn = CropBrainRegion()
        random.seed(7)
        fig, axes = plt.subplots(2, len(class_names), figsize=(4 * len(class_names), 8))
        for col, cls in enumerate(class_names):
            cls_dir = os.path.join(TEST_DIR, cls)
            fname = random.choice(os.listdir(cls_dir))
            img_path = os.path.join(cls_dir, fname)
            input_tensor, display_img = load_image_for_gradcam(img_path, crop_fn, IMG_SIZE)
            input_tensor = input_tensor.to(device)
            cam, conf, output = gradcam.generate(input_tensor, class_names.index(cls))
            pred_idx = output.argmax(dim=1).item()
            overlay = overlay_heatmap(display_img, cam)

            axes[0, col].imshow(display_img)
            axes[0, col].set_title(f"True: {cls}")
            axes[0, col].axis("off")
            axes[1, col].imshow(overlay)
            axes[1, col].set_title(f"Pred: {class_names[pred_idx]} ({conf*100:.1f}%)")
            axes[1, col].axis("off")
        gradcam.remove_hooks()
        fig.suptitle("Grad-CAM: Original MRI (top) vs. Model Attention Heatmap (bottom)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "gradcam_examples.png"), dpi=150)
        plt.close(fig)
        print(f"Saved Grad-CAM example grid to {os.path.join(OUT_DIR, 'gradcam_examples.png')}")
    except Exception as e:
        print(f"Grad-CAM example generation failed (non-fatal): {e}")

    total_time = time.time() - start_time
    print(f"\nALL DONE in {total_time/60:.1f} minutes. "
          f"Final test accuracy: {report['accuracy']*100:.2f}% (TTA: {test_acc_tta*100:.2f}%)")


if __name__ == "__main__":
    main()
