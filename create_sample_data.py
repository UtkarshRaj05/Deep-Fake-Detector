"""
Generates a small synthetic dataset so you can run train.py / evaluate.py
/ the web app end-to-end immediately, before plugging in a real deepfake
dataset. These images are NOT real faces and the resulting model has NO
real-world detection ability -- this script exists purely to exercise
the pipeline (data loading, training loop, checkpointing, inference,
Grad-CAM) and catch bugs early.

For real training, point config.DATASET_DIR at a dataset such as:
  - FaceForensics++ (https://github.com/ondyari/FaceForensics)
  - Celeb-DF
  - DFDC (Deepfake Detection Challenge)
with frames extracted and organized into train/val/test x real/fake folders.

Usage:
    python create_sample_data.py
"""
import os
import random

import numpy as np
from PIL import Image, ImageDraw

import config

random.seed(42)
np.random.seed(42)

N_PER_SPLIT = {"train": 40, "val": 12, "test": 12}
IMG_SIZE = 256


def draw_synthetic_face(is_fake: bool) -> Image.Image:
    """
    Draws a crude cartoon 'face' with random noise characteristics.
    Fake examples get extra high-frequency noise + asymmetry to give the
    model *something* learnable, purely so the training loop has signal
    to converge on during a pipeline smoke test.
    """
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=tuple(np.random.randint(160, 220, 3)))
    draw = ImageDraw.Draw(img)

    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    face_r = 90
    skin = tuple(np.random.randint(180, 235, 3))
    draw.ellipse([cx - face_r, cy - face_r, cx + face_r, cy + face_r], fill=skin)

    eye_dx = 30 if not is_fake else 30 + random.randint(-8, 8)
    eye_y = cy - 20
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        draw.ellipse([ex - 12, eye_y - 8, ex + 12, eye_y + 8], fill=(255, 255, 255))
        draw.ellipse([ex - 5, eye_y - 5, ex + 5, eye_y + 5], fill=(40, 40, 40))

    draw.arc([cx - 30, cy + 10, cx + 30, cy + 45], start=20, end=160, fill=(120, 60, 60), width=4)

    arr = np.array(img).astype(np.float32)
    noise_level = 18 if is_fake else 6
    noise = np.random.normal(0, noise_level, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    if is_fake:
        # simulate a soft blending-seam artifact around the face boundary
        seam = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        sd = ImageDraw.Draw(seam)
        sd.ellipse([cx - face_r - 4, cy - face_r - 4, cx + face_r + 4, cy + face_r + 4], outline=255, width=6)
        seam_arr = np.array(seam)
        arr[seam_arr > 0] = np.clip(arr[seam_arr > 0].astype(int) + random.randint(15, 40), 0, 255)

    return Image.fromarray(arr)


def generate_split(split: str, n: int):
    for label, is_fake in (("real", False), ("fake", True)):
        out_dir = os.path.join(config.DATASET_DIR, split, label)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(n):
            img = draw_synthetic_face(is_fake)
            img.save(os.path.join(out_dir, f"{label}_{i:03d}.png"))
        print(f"  {split}/{label}: {n} images")


def main():
    print(f"Generating synthetic sample dataset under {config.DATASET_DIR} ...")
    for split, n in N_PER_SPLIT.items():
        generate_split(split, n)
    print("\nDone. This is placeholder data for pipeline testing only -- ")
    print("swap in a real dataset (FaceForensics++, Celeb-DF, DFDC, etc.) before relying on results.")


if __name__ == "__main__":
    main()
