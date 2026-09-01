"""
Central configuration for the deepfake image detection system.
Adjust paths and hyperparameters here rather than scattering magic
numbers through the codebase.
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data ---
DATASET_DIR = os.path.join(BASE_DIR, "sample_dataset")  # point this at your real dataset
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMAGE_SIZE = 224          # input resolution the backbone expects
FACE_MARGIN = 0.35        # extra crop margin around detected face, as fraction of face box

# --- Model ---
BACKBONE = "efficientnet_b0"
NUM_CLASSES = 2           # 0 = real, 1 = fake
CLASS_NAMES = ["real", "fake"]

# --- Training ---
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE_EPOCHS = 2   # train only the head for this many epochs first
EARLY_STOP_PATIENCE = 5
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Checkpoints ---
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pt")

# --- Web app ---
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_UPLOAD_MB = 12

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
