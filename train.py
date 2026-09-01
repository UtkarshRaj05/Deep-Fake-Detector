"""
Train the deepfake detector.

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 32 --data-dir /path/to/dataset

Expects a dataset directory with train/ and val/ subfolders, each
containing real/ and fake/ image folders (see data/dataset.py).
"""
import argparse
import time
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data.dataset import DeepfakeDataset
from models.detector import DeepfakeDetector


def parse_args():
    p = argparse.ArgumentParser(description="Train the deepfake detector")
    p.add_argument("--data-dir", default=config.DATASET_DIR)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--freeze-epochs", type=int, default=config.FREEZE_BACKBONE_EPOCHS)
    p.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    p.add_argument("--no-pretrained", action="store_true",
                    help="Train the backbone from random init (no internet needed)")
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    print(f"Device: {config.DEVICE}")
    print(f"Loading data from {args.data_dir} ...")

    train_ds = DeepfakeDataset(train_dir, config.IMAGE_SIZE, train=True)
    val_ds = DeepfakeDataset(val_dir, config.IMAGE_SIZE, train=False)

    n_real, n_fake = train_ds.class_balance()
    print(f"Train set: {len(train_ds)} images ({n_real} real / {n_fake} fake)")
    print(f"Val set:   {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    model = DeepfakeDetector(num_classes=config.NUM_CLASSES,
                              pretrained=not args.no_pretrained).to(config.DEVICE)

    # class-weighted loss in case the dataset is imbalanced (common in
    # deepfake datasets, which often have far more fake than real frames)
    class_weights = None
    if n_real > 0 and n_fake > 0:
        total = n_real + n_fake
        class_weights = torch.tensor(
            [total / (2 * n_real), total / (2 * n_fake)], dtype=torch.float32
        ).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.set_backbone_trainable(epoch > args.freeze_epochs)
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, config.DEVICE, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, config.DEVICE, train=False)
        scheduler.step(val_loss)

        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.3f} | {dt:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        torch.save({"model_state": model.state_dict(), "epoch": epoch}, config.LAST_MODEL_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                        "val_acc": val_acc}, config.BEST_MODEL_PATH)
            print(f"  -> new best model saved (val_loss {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    with open(os.path.join(config.CHECKPOINT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best model: {config.BEST_MODEL_PATH} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
