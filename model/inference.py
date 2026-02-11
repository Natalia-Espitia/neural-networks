import io
import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


class SimpleCNN(tf.keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.fc2 = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)


def _load_classes(model_dir: Path):
    classes_file = model_dir / "classes.txt"
    if classes_file.exists():
        classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if classes:
            return classes
    return ["cat", "dog"]


def _prepare_image(payload, img_size=128):
    if isinstance(payload, np.ndarray):
        arr = payload.astype(np.float32)
    else:
        arr = np.asarray(payload, dtype=np.float32)

    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    arr = tf.image.resize(arr, [img_size, img_size]).numpy()
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr.astype(np.float32)


def model_fn(model_dir):
    model_dir = Path(model_dir)
    classes = _load_classes(model_dir)

    model = SimpleCNN(num_classes=len(classes))
    model.build((None, 128, 128, 3))

    with (model_dir / "model.pth").open("rb") as f:
        weights = pickle.load(f)
    model.set_weights(weights)

    return {"model": model, "classes": classes}


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        return np.array(payload.get("instances", payload), dtype=np.float32)

    if request_content_type in ("image/jpeg", "image/jpg", "image/png"):
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        return np.array(image, dtype=np.float32)

    raise ValueError(f"Content-Type no soportado: {request_content_type}")


def predict_fn(input_data, model_bundle):
    model = model_bundle["model"]
    classes = model_bundle["classes"]

    x = _prepare_image(input_data, img_size=128)
    logits = model(x, training=False).numpy()
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    idx = int(np.argmax(probs))

    return {
        "predicted_index": idx,
        "predicted_class": classes[idx],
        "probabilities": probs.tolist()
    }


def output_fn(prediction, accept):
    if accept in ("application/json", "*/*"):
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Accept no soportado: {accept}")
