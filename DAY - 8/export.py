"""Step 5a: export the trained model for deployment.

Saves the model in the native .keras format (recommended) and, optionally, as a
TensorFlow SavedModel directory for serving.

Usage:
    python export.py
    python export.py --saved-model
"""

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser(description="Export the trained CIFAR-10 model.")
    p.add_argument("--model", default="outputs/augmented/best_model.keras")
    p.add_argument("--output-dir", default="exported_model")
    p.add_argument("--saved-model", action="store_true",
                   help="also export a TensorFlow SavedModel directory (for TF Serving)")
    args = p.parse_args()

    from tensorflow import keras

    if not os.path.exists(args.model):
        raise SystemExit(f"error: model not found: {args.model}. Run train.py first.")
    os.makedirs(args.output_dir, exist_ok=True)

    model = keras.models.load_model(args.model)

    keras_path = os.path.join(args.output_dir, "cifar10_cnn.keras")
    model.save(keras_path)
    print(f"Saved {keras_path}")

    if args.saved_model:
        sm_path = os.path.join(args.output_dir, "saved_model")
        model.export(sm_path)
        print(f"Saved TensorFlow SavedModel to {sm_path}")

    print(f"Classify an image with: python predict.py --model {keras_path} path/to/image.jpg")


if __name__ == "__main__":
    main()
