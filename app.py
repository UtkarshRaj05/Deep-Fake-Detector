"""
Web app: upload an image, get a real/fake verdict with confidence and a
Grad-CAM heatmap showing which regions drove the decision.

Usage:
    python app.py
Then open http://localhost:5000
"""
import os
import base64
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

import config
from inference import DeepfakeInference

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

print("Loading model...")
engine = DeepfakeInference()
print(f"Model ready. Trained checkpoint loaded: {engine.checkpoint_loaded}")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", checkpoint_loaded=engine.checkpoint_loaded)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, WEBP or BMP."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read that file as an image."}), 400

    # thumbnail of the original, sent back so the UI can show it alongside the heatmap
    original = image.copy()
    original.thumbnail((900, 900))
    buf = BytesIO()
    original.save(buf, format="PNG")
    original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    result = engine.predict(image, with_heatmap=True)
    result["original_base64"] = original_b64
    result["filename"] = secure_filename(file.filename)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
