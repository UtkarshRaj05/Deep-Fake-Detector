# Aperture — Deepfake Image Detection System

A complete, working pipeline for detecting AI-manipulated (deepfake) face
images: face-aware preprocessing, a transfer-learning CNN, Grad-CAM
interpretability, a training/evaluation harness, a CLI, and a web app.

```
Upload image → detect & crop face → CNN classifier → real/fake + confidence
                                                     → Grad-CAM heatmap
```

## What's included

```
deepfake-detector/
├── config.py              # all paths & hyperparameters in one place
├── data/dataset.py        # dataset loader (expects real/ + fake/ folders)
├── models/detector.py     # EfficientNet-B0 backbone + classifier head
├── utils/face_utils.py    # face detection & cropping (OpenCV Haar cascade)
├── utils/grad_cam.py      # Grad-CAM heatmap generation
├── inference.py           # shared prediction pipeline (used by CLI + web app)
├── train.py                # training loop: class-weighted loss, LR scheduling,
│                            #   backbone freezing, early stopping, checkpointing
├── evaluate.py             # accuracy / precision / recall / F1 / confusion matrix
├── predict.py               # CLI: `python predict.py image.jpg`
├── app.py                   # Flask web app
├── templates/index.html, static/style.css, static/script.js
├── create_sample_data.py    # generates a small synthetic dataset for smoke-testing
└── checkpoints/             # trained weights land here
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate placeholder data so you can exercise the full pipeline immediately
#    (synthetic cartoon faces — NOT real training data, see warning below)
python create_sample_data.py

# 2. Train
python train.py

# 3. Check metrics
python evaluate.py

# 4. Try the CLI
python predict.py sample_dataset/test/fake/fake_000.png

# 5. Launch the web app
python app.py
# open http://localhost:5000
```

## Using a real dataset

`create_sample_data.py` only exists so every part of the pipeline (data
loading, training loop, checkpointing, inference, Grad-CAM) is exercised
and bug-free before you touch real data — the synthetic images have no
relationship to real deepfake artifacts and a model trained only on them
has **no real-world detection ability**.

To train for real, get a public deepfake dataset, extract frames, and
organize them like this:

```
your_dataset/
├── train/
│   ├── real/   *.jpg
│   └── fake/   *.jpg
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

Good starting points:
- **FaceForensics++** — github.com/ondyari/FaceForensics
- **Celeb-DF (v2)**
- **DFDC** (Deepfake Detection Challenge, via Kaggle)

Then either edit `DATASET_DIR` in `config.py`, or pass `--data-dir`:

```bash
python train.py --data-dir /path/to/your_dataset --epochs 20 --batch-size 32
```

A few thousand images per class is a reasonable starting point; more (and
more *diverse* sources/generators) will generalize better. Real deepfake
datasets are usually heavily imbalanced — `train.py` already applies
class-weighted loss to compensate.

## How the model works

1. **Face crop.** `utils/face_utils.py` runs OpenCV's Haar cascade face
   detector and crops to the largest face with a margin, since manipulation
   artifacts (blending seams, warped features, inconsistent skin texture)
   concentrate around the face. Falls back to the full image if no face is
   found.
2. **Classification.** `models/detector.py` uses an EfficientNet-B0 backbone
   (ImageNet-pretrained when a network connection is available, otherwise a
   random init so the code never hard-fails offline) with a small
   dropout + linear classification head, fine-tuned end to end.
3. **Interpretability.** `utils/grad_cam.py` implements Grad-CAM against the
   backbone's last convolutional layer, producing a heatmap of which pixels
   most influenced the verdict — shown as a toggle in the web app.

## Notes & limitations

- This is a research/education-grade pipeline, not a production-hardened
  forensic tool. No detector generalizes perfectly to *generator
  architectures it wasn't trained on* — a model trained only on one dataset's
  fakes will do worse against images from a newer/different generator.
- Treat output as one signal among several, not a courtroom-ready verdict.
- The web app's dev server (`python app.py`) is for local/demo use. For
  anything public-facing, run it behind a real WSGI server (gunicorn/uWSGI)
  and add authentication/rate limiting.
