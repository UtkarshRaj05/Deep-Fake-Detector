"""
Face detection and cropping utilities.

Deepfake artifacts are concentrated around the face (blending boundaries,
warped features, inconsistent skin texture), so we detect and crop the
face before classification rather than feeding the whole image to the
network. Falls back gracefully to the full image if no face is found,
so the pipeline never hard-fails on a difficult image.
"""
import cv2
import numpy as np
from PIL import Image

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_detector = cv2.CascadeClassifier(_CASCADE_PATH)


def detect_largest_face(image_bgr: np.ndarray):
    """Return (x, y, w, h) of the largest detected face, or None."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    # pick the largest face by area (most likely the subject)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def crop_face(pil_image: Image.Image, margin: float = 0.35):
    """
    Detect the largest face in a PIL image and return a cropped PIL image
    with a margin around it. Returns (cropped_image, face_found: bool,
    box_in_original_coords or None).
    """
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    box = detect_largest_face(bgr)

    h_img, w_img = bgr.shape[:2]

    if box is None:
        return pil_image, False, None

    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w_img, x + w + mx)
    y1 = min(h_img, y + h + my)

    cropped = pil_image.crop((x0, y0, x1, y1))
    return cropped, True, (x0, y0, x1, y1)
