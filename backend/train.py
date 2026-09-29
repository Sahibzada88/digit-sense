"""
train.py — end-to-end training pipeline for DigitCNN v2, on the full
MNIST dataset.

Pipeline
--------
  1. Load the real MNIST dataset (50k train / 10k val / 10k test,
     28x28 grayscale) from data/mnist.pkl.gz.
  2. Apply on-the-fly data augmentation to the training split (random
     rotation, translation, and slight scaling) so the model is robust
     to imperfectly drawn freehand digits from the web canvas.
  3. Train the residual CNN with Adam + cosine-annealed learning rate,
     tracking train/val loss and accuracy every epoch, with early
     stopping and best-checkpoint selection on validation accuracy.
  4. Evaluate on the held-out test split; save a classification report,
     a confusion matrix plot, and a loss/accuracy curve plot to
     artifacts/ for the README / analytics dashboard.
  5. Save the best weights to saved_model/digit_cnn.pth and a metadata
     JSON describing the run (architecture, accuracy, param count,
     timestamp) for the API's /model-info endpoint.

Run:  python3 train.py [--epochs 18] [--batch-size 128]
"""

import argparse
import gzip
import json
import os
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import DigitCNN

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "mnist.pkl.gz")
SAVE_DIR = os.path.join(HERE, "saved_model")
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data ----

def load_mnist():
    with gzip.open(DATA_PATH, "rb") as f:
        train, val, test = pickle.load(f, encoding="latin1")
    # Each split is (flat_images[N,784] in [0,1], labels[N])
    return train, val, test


class MNISTDataset(Dataset):
    """Wraps flat MNIST arrays as 28x28 tensors, with optional light
    geometric augmentation applied per-sample at load time."""

    def __init__(self, images, labels, augment: bool = False):
        self.images = images.reshape(-1, 28, 28).astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.augment:
            angle = np.random.uniform(-12, 12)
            tx, ty = np.random.uniform(-2, 2, size=2)
            scale = np.random.uniform(0.9, 1.1)
            center = (14, 14)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (28, 28), borderValue=0.0)

            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 0.03, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0, 1)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,28,28)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.long)


# ------------------------------------------------------------- training ----

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def plot_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — MNIST test split")
    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    train_raw, val_raw, test_raw = load_mnist()

    train_ds = MNISTDataset(*train_raw, augment=True)
    val_ds = MNISTDataset(*val_raw, augment=False)
    test_ds = MNISTDataset(*test_raw, augment=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = DigitCNN().to(DEVICE)
    print(f"Model parameters: {model.num_parameters():,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_without_improvement = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t_epoch = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "digit_cnn.pth"))
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{time.time() - t_epoch:.1f}s"
            f"{'  * best' if improved else ''}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping — no val improvement in {args.patience} epochs.")
            break

    total_time = time.time() - t_start
    print(f"\nTraining finished in {total_time/60:.1f} min. Best val accuracy: {best_val_acc:.4f}")

    # ---- Final evaluation on held-out test set with the best checkpoint ----
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "digit_cnn.pth")))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"Test accuracy: {test_acc:.4f}")
    report = classification_report(labels, preds, digits=3)
    print(report)

    cm = confusion_matrix(labels, preds)

    # ---- Artifacts: plots + metadata for README / dashboard ----
    plot_curves(history, os.path.join(ARTIFACT_DIR, "training_curves.png"))
    plot_confusion(cm, os.path.join(ARTIFACT_DIR, "confusion_matrix.png"))

    with open(os.path.join(ARTIFACT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    with open(os.path.join(ARTIFACT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    metadata = {
        "architecture": "DigitCNN v2 — stem + 3 residual stages (32->64->128->256 ch)",
        "framework": f"PyTorch {torch.__version__}",
        "dataset": "Full MNIST (50k train / 10k val / 10k test)",
        "parameters": model.num_parameters(),
        "epochs_trained": len(history["train_loss"]),
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "training_time_minutes": round(total_time / 60, 2),
        "device": str(DEVICE),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved weights -> {os.path.join(SAVE_DIR, 'digit_cnn.pth')}")
    print(f"Saved artifacts -> {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
