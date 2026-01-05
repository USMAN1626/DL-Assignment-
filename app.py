import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, static_folder="frontend", template_folder="frontend")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload cap


def load_vgg_model():
    model_path = os.getenv("VGG_MODEL_PATH", os.path.join("models", "vgg16_model.h5"))
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            source = f"loaded from {model_path}"
            return model, source
        except Exception:
            
            pass

    base = tf.keras.applications.vgg16.VGG16(weights="imagenet")
    model = tf.keras.Model(inputs=base.inputs, outputs=base.layers[-2].output)
    source = "imagenet weights (fallback)"
    return model, source


model, model_source = load_vgg_model()


def preprocess_image(file_storage):
    image_bytes = file_storage.read()
    image = tf.keras.utils.load_img(BytesIO(image_bytes), target_size=(224, 224))
    tensor = tf.keras.utils.img_to_array(image)
    tensor = np.expand_dims(tensor, axis=0)
    tensor = tf.keras.applications.vgg16.preprocess_input(tensor)
    return tensor


@app.route("/")
def index():
    return render_template("index.html", model_source=model_source)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_source": model_source})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    file_storage = request.files["image"]
    if file_storage.filename == "":
        return jsonify({"error": "image file is required"}), 400

    try:
        input_tensor = preprocess_image(file_storage)
        embedding = model.predict(input_tensor, verbose=0)
    except Exception as exc: 
        return jsonify({"error": f"failed to generate embedding: {exc}"}), 500

    flat = embedding.flatten()
    preview = flat[:10].round(4).tolist()
    stats = {
        "shape": list(embedding.shape),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }

    return jsonify({
        "preview": preview,
        "statistics": stats,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
