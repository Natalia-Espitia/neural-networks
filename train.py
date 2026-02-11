import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "."))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts and p.is_file()])


def build_dataset(training_root: Path, img_size: int, batch_size: int):
    candidate_roots = [training_root, Path.cwd(), Path(__file__).resolve().parent]

    cats_dir = None
    dogs_dir = None

    for root in candidate_roots:
        direct_cats = root / "cats_set"
        direct_dogs = root / "dogs_set"
        if direct_cats.exists() and direct_dogs.exists():
            cats_dir, dogs_dir = direct_cats, direct_dogs
            break

        arch_cats = root / "archive" / "cats_set"
        arch_dogs = root / "archive" / "dogs_set"
        if arch_cats.exists() and arch_dogs.exists():
            cats_dir, dogs_dir = arch_cats, arch_dogs
            break

    if cats_dir is None or dogs_dir is None:
        raise FileNotFoundError(
            "No se encontraron cats_set y dogs_set. "
            "Estructuras soportadas: <training>/cats_set y <training>/dogs_set, "
            "o <training>/archive/cats_set y <training>/archive/dogs_set."
        )

    cat_files = list_images(cats_dir)
    dog_files = list_images(dogs_dir)
    if not cat_files or not dog_files:
        raise ValueError("No hay imagenes en cats_set o dogs_set.")

    paths = np.array([str(p) for p in (cat_files + dog_files)])
    labels = np.array([0] * len(cat_files) + [1] * len(dog_files))

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(paths))
    paths = paths[idx]
    labels = labels[idx]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    args = parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    train_ds = build_dataset(Path(args.training), args.img_size, args.batch_size)

    model = SimpleCNN(num_classes=2)
    model.build((None, args.img_size, args.img_size, 3))

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=criterion, metrics=["accuracy"])

    history = model.fit(train_ds, epochs=args.epochs, verbose=1)

    for epoch_idx in range(args.epochs):
        loss_value = history.history["loss"][epoch_idx]
        print(f"Epoch {epoch_idx + 1}, Loss: {loss_value:.4f}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.pth").open("wb") as f:
        pickle.dump(model.get_weights(), f)

    print(f"Pesos exportados en: {model_dir / 'model.pth'}")


if __name__ == "__main__":
    main()
