"""
Shared inference pipeline: load model once, run face-crop -> classify ->
Grad-CAM on any input image. Both predict.py (CLI) and app.py (web) call
into this module so the two interfaces never drift apart.
"""
import os
import base64
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image

import config
from models.detector import DeepfakeDetector
from data.dataset import build_transforms
from utils.face_utils import crop_face
from utils.grad_cam import GradCAM, overlay_heatmap


class DeepfakeInference:
    def __init__(self, checkpoint_path: str = config.BEST_MODEL_PATH):
        self.device = config.DEVICE
        self.model = DeepfakeDetector(num_classes=config.NUM_CLASSES, pretrained=False).to(self.device)

        self.checkpoint_loaded = False
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.checkpoint_loaded = True

        self.model.eval()
        self.transform = build_transforms(config.IMAGE_SIZE, train=False)
        self.gradcam = GradCAM(self.model, self.model.gradcam_target_layer)

    def predict(self, pil_image: Image.Image, with_heatmap: bool = True) -> dict:
        """
        Run the full pipeline on a single PIL image.
        Returns a dict with the verdict, confidence, face-detection info,
        and (optionally) a base64-encoded Grad-CAM overlay image.
        """
        cropped, face_found, box = crop_face(pil_image, margin=config.FACE_MARGIN)
        analyzed_image = cropped if face_found else pil_image

        input_tensor = self.transform(analyzed_image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]

        fake_prob = probs[1].item()
        real_prob = probs[0].item()
        pred_class = 1 if fake_prob >= real_prob else 0

        result = {
            "label": config.CLASS_NAMES[pred_class],
            "confidence": max(fake_prob, real_prob),
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "face_detected": face_found,
            "checkpoint_loaded": self.checkpoint_loaded,
            "heatmap_base64": None,
        }

        if with_heatmap:
            cam = self.gradcam.generate(input_tensor, class_idx=pred_class)
            overlay = overlay_heatmap(analyzed_image, cam)
            result["heatmap_base64"] = _pil_to_base64(Image.fromarray(overlay))

        return result


def _pil_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
