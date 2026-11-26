"""Step 2: CNN architecture with Conv2D, MaxPooling and Dropout layers."""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(32, 32, 3), num_classes=10) -> keras.Model:
    """Three conv blocks (Conv-Conv-Pool-Dropout) followed by a dense classifier."""
    inputs = keras.Input(shape=input_shape)

    # Light augmentation: only active during training
    x = layers.RandomFlip("horizontal")(inputs)
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
