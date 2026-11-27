"""CNN architecture: Conv2D, BatchNorm, MaxPooling2D and Dropout layers."""

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape: Tuple[int, int, int] = (32, 32, 3), num_classes: int = 10,
              builtin_augmentation: bool = False) -> keras.Model:
    """Three conv blocks (Conv-Conv-Pool-Dropout) followed by a dense classifier.

    builtin_augmentation: the original version had Keras flip/translate layers
    inside the model. They are OFF by default now, because augmentation is done
    by the imgaug tf.data pipeline (and the baseline must have none). Turn on
    only if you want Keras-layer augmentation instead of imgaug.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    if builtin_augmentation:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomTranslation(0.1, 0.1)(x)

    for filters, drop in [(32, 0.25), (64, 0.25), (128, 0.40)]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(drop)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cifar10_cnn")


if __name__ == "__main__":
    build_cnn().summary()
