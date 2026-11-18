"""
model.py - Convolutional Neural Network (CNN) for Face Emotion Recognition
College Project: Face Emotion Recognition Using CNN
Subject: Advanced Machine Learning and Deep Learning

Architecture:
- Input layer: 48x48x1 (grayscale face image)
- 4 Feature Extraction Blocks:
    Block 1: Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
    Block 2: Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
    Block 3: Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
    Block 4: Conv2D(256) + BatchNorm + MaxPool + Dropout(0.25)
- Classification Head:
    Flatten + Dense(512) + BatchNorm + Dropout(0.5) + Dense(256) + Dropout(0.3) + Dense(7, Softmax)
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES, LEARNING_RATE, MODEL_SUMMARY_PATH


def build_emotion_model():
    """
    Builds and compiles the 4-block CNN model for facial emotion recognition.
    Each block extracts increasingly complex facial features (edges -> textures -> parts -> expressions).
    """
    model = models.Sequential(name="Face_Emotion_CNN")

    # ==========================================
    # Convolutional Block 1: Low-level features (edges, contours)
    # 32 filters of size 3x3 with ReLU activation and 'same' padding
    # ==========================================
    model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"))
    model.add(layers.BatchNormalization(name="bn1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))
    model.add(layers.Dropout(0.25, name="dropout1"))

    # ==========================================
    # Convolutional Block 2: Textures & shapes
    # 64 filters of size 3x3 to capture richer facial patterns
    # ==========================================
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"))
    model.add(layers.BatchNormalization(name="bn2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))
    model.add(layers.Dropout(0.25, name="dropout2"))

    # ==========================================
    # Convolutional Block 3: Facial components (eyes, nose, mouth corners)
    # 128 filters of size 3x3
    # ==========================================
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"))
    model.add(layers.BatchNormalization(name="bn3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))
    model.add(layers.Dropout(0.25, name="dropout3"))

    # ==========================================
    # Convolutional Block 4: Complex expression features
    # 256 filters of size 3x3 for high-level semantic representation
    # ==========================================
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4"))
    model.add(layers.BatchNormalization(name="bn4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool4"))
    model.add(layers.Dropout(0.25, name="dropout4"))

    # ==========================================
    # Fully-Connected (Dense) Classification Head
    # Flattens 2D feature maps to 1D vector and classifies into 7 emotions
    # ==========================================
    model.add(layers.Flatten(name="flatten"))

    # Dense layer 1: 512 neurons with Batch Normalization & 50% Dropout for regularization
    model.add(layers.Dense(512, activation="relu", name="dense1"))
    model.add(layers.BatchNormalization(name="bn_dense"))
    model.add(layers.Dropout(0.5, name="dropout_dense1"))

    # Dense layer 2: 256 neurons with 30% Dropout
    model.add(layers.Dense(256, activation="relu", name="dense2"))
    model.add(layers.Dropout(0.3, name="dropout_dense2"))

    # Output layer: 7 neurons with Softmax activation (one probability per emotion)
    model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="output_emotions"))

    # Compile model with Adam optimizer and Categorical Cross-Entropy Loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_model_summary(model, output_path=MODEL_SUMMARY_PATH):
    """
    Prints model summary to console and writes it to a text file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    summary_text = "\n".join(summary_lines)

    print("\n" + "=" * 60)
    print("CNN ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(summary_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Face Emotion Recognition CNN Model Summary\n")
        f.write("=" * 60 + "\n")
        f.write(summary_text + "\n")

    print(f"\n[OK] Model summary saved to: {output_path}")


if __name__ == "__main__":
    cnn_model = build_emotion_model()
    save_model_summary(cnn_model)
