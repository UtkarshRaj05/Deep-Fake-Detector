from pathlib import Path
import cv2
import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"

_face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

if _face_detector.empty():
    raise RuntimeError(f"Could not load Haar Cascade: {CASCADE_PATH}")


def detect_largest_face(image_bgr: np.ndarray):
    """Return the largest face coordinates or None."""

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = _face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    return int(x), int(y), int(w), int(h)


def crop_face(
    pil_image: Image.Image,
    margin: float = 0.35
):
    """Detect and crop the largest face."""

    image_rgb = np.array(
        pil_image.convert("RGB")
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    face = detect_largest_face(image_bgr)

    if face is None:
        return pil_image, False, None

    x, y, w, h = face

    img_h, img_w = image_rgb.shape[:2]

    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    cropped = pil_image.crop(
        (x1, y1, x2, y2)
    )

    return cropped, True, (x1, y1, x2, y2)
