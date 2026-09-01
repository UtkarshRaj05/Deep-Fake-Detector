"""
CLI: run the deepfake detector on one or more images.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/folder/ --save-heatmaps out/
"""
import argparse
import base64
import os

from PIL import Image

import config
from inference import DeepfakeInference

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    p = argparse.ArgumentParser(description="Detect whether image(s) are AI-manipulated")
    p.add_argument("path", help="Path to an image file or a folder of images")
    p.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    p.add_argument("--save-heatmaps", default=None,
                    help="Optional directory to save Grad-CAM heatmap overlays to")
    return p.parse_args()


def gather_images(path):
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(IMG_EXTS)]
    return [path]


def main():
    args = parse_args()
    engine = DeepfakeInference(checkpoint_path=args.checkpoint)

    if not engine.checkpoint_loaded:
        print(f"NOTE: no trained checkpoint found at {args.checkpoint}.")
        print("Predictions below use an untrained/ImageNet-only backbone and are not meaningful.")
        print("Run `python train.py` first on a labeled dataset.\n")

    if args.save_heatmaps:
        os.makedirs(args.save_heatmaps, exist_ok=True)

    images = gather_images(args.path)
    if not images:
        print(f"No images found at {args.path}")
        return

    for img_path in images:
        image = Image.open(img_path)
        result = engine.predict(image, with_heatmap=bool(args.save_heatmaps))

        verdict = result["label"].upper()
        conf = result["confidence"] * 100
        face_note = "face detected" if result["face_detected"] else "no face found, analyzed full image"
        print(f"{os.path.basename(img_path):40s} -> {verdict:5s} ({conf:.1f}% confidence, {face_note})")

        if args.save_heatmaps and result["heatmap_base64"]:
            out_path = os.path.join(args.save_heatmaps, f"heatmap_{os.path.basename(img_path)}.png")
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(result["heatmap_base64"]))


if __name__ == "__main__":
    main()
