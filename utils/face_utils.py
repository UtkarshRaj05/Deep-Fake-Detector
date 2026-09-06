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


# Use OpenCV's built-in Haar Cascade path
_CASCADE_PATH = (
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_face_detector = cv2.CascadeClassifier(_CASCADE_PATH)

# Check if cascade loaded correctly
if _face_detector.empty():
    raise RuntimeError(
        f"Failed to load Haar Cascade classifier: {_CASCADE_PATH}"
    )


def detect_largest_face(image_bgr: np.ndarray):
    """Return (x, y, w, h) of the largest detected face, or None."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    # Pick the largest face
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    return int(x), int(y), int(w), int(h)


def crop_face(
    pil_image: Image.Image,
    margin: float = 0.35
):
    """
    Detect the largest face in a PIL image and return a cropped PIL image.

    Returns:
        cropped_image,
        face_found,
        box_in_original_coords
    """

    # Convert PIL image to NumPy array
    image_rgb = np.array(pil_image.convert("RGB"))

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    face = detect_largest_face(image_bgr)

    # If no face is detected, return original image
    if face is None:
        return pil_image, False, None

    x, y, w, h = face

    # Add margin around detected face
    img_h, img_w = image_rgb.shape[:2]

    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    # Crop the face
    cropped = pil_image.crop((x1, y1, x2, y2))

    return cropped, True, (x1, y1, x2, y2)
