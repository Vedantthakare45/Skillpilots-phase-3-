"""Step 5b: inference script. Classify your own images with the exported model.

Usage:
    python predict.py photo.jpg
    python predict.py cat.png dog.png --top 3
    python predict.py --model exported_model/cifar10_cnn.keras photo.jpg
"""

import argparse
import os

import numpy as np
from PIL import Image

from data import CLASS_NAMES

IMG_SIZE = (32, 32)


def preprocess_image(path: str) -> np.ndarray:
    """Load any image file -> float32 array of shape (1, 32, 32, 3) scaled to [0, 1].

    Handles grayscale, RGBA and palette images by converting to RGB first.
    """
    with Image.open(path) as img:
        img = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
        arr = np.asarray(img, dtype="float32") / 255.0
    return arr[np.newaxis, ...]


def top_k(probs: np.ndarray, k: int = 3):
    """Return [(class_name, probability), ...] for the k most likely classes."""
    k = max(1, min(k, len(CLASS_NAMES)))
    idx = np.argsort(probs)[::-1][:k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in idx]


def main():
    p = argparse.ArgumentParser(description="Classify images with the trained CIFAR-10 CNN.")
    p.add_argument("images", nargs="+", help="one or more image files")
    p.add_argument("--model", default="exported_model/cifar10_cnn.keras")
    p.add_argument("--top", type=int, default=3, help="how many top classes to show")
    args = p.parse_args()

    from tensorflow import keras

    if not os.path.exists(args.model):
        raise SystemExit(f"error: model not found: {args.model}. Run train.py and export.py first.")
    model = keras.models.load_model(args.model)

    for path in args.images:
        if not os.path.exists(path):
            print(f"{path}: file not found")
            continue
        probs = model.predict(preprocess_image(path), verbose=0)[0]
        print(f"\n{path}")
        for name, prob in top_k(probs, args.top):
            print(f"  {name:<11s} {prob:6.1%}")


if __name__ == "__main__":
    main()
