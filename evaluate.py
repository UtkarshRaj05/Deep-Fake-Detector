"""
Evaluate a trained checkpoint on the test set: accuracy, precision,
recall, F1, and a confusion matrix.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pt --data-dir /path/to/dataset
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

import config
from data.dataset import DeepfakeDataset
from models.detector import DeepfakeDetector


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the deepfake detector")
    p.add_argument("--data-dir", default=config.DATASET_DIR)
    p.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()
    test_dir = os.path.join(args.data_dir, "test")

    test_ds = DeepfakeDataset(test_dir, config.IMAGE_SIZE, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=config.NUM_WORKERS)

    model = DeepfakeDetector(num_classes=config.NUM_CLASSES, pretrained=False).to(config.DEVICE)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"WARNING: no checkpoint found at {args.checkpoint}. "
              f"Evaluating an untrained model -- results are meaningless "
              f"until you run train.py.")

    model.eval()

    tp = tn = fp = fn = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            preds = model(images).argmax(dim=1)

            for p, l in zip(preds.tolist(), labels.tolist()):
                if p == 1 and l == 1:
                    tp += 1
                elif p == 0 and l == 0:
                    tn += 1
                elif p == 1 and l == 0:
                    fp += 1
                elif p == 0 and l == 1:
                    fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print("\n--- Test results ---")
    print(f"Samples:    {total}")
    print(f"Accuracy:   {accuracy:.3f}")
    print(f"Precision:  {precision:.3f}  (of images flagged fake, how many really were)")
    print(f"Recall:     {recall:.3f}  (of real fakes, how many were caught)")
    print(f"F1:         {f1:.3f}")
    print("\nConfusion matrix:")
    print(f"                 pred_real   pred_fake")
    print(f"  actual_real    {tn:9d}   {fp:9d}")
    print(f"  actual_fake    {fn:9d}   {tp:9d}")


if __name__ == "__main__":
    main()
