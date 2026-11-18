"""
train.py - Training Pipeline with Data Augmentation & Class Imbalance Handling
College Project: Face Emotion Recognition Using CNN
Subject: Advanced Machine Learning and Deep Learning

Trains the 4-block CNN on FER2013 with:
- ImageDataGenerator (grayscale, normalized, augmented on train only)
- Balanced class weights computed via scikit-learn
- ModelCheckpoint (best val_accuracy saved to outputs/emotion_model.keras)
- ReduceLROnPlateau learning rate scheduler
- Strict cap: Maximum 5 epochs allowed per project requirements
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from config import (
    TRAIN_DIR,
    TEST_DIR,
    MODEL_PATH,
    HISTORY_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    MAX_EPOCHS,
    EMOTIONS,
)
from model import build_emotion_model, save_model_summary


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Face Emotion Recognition CNN")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs (capped strictly at 5)",
    )
    args = parser.parse_args()

    # Rule: Training must NEVER run more than 5 epochs
    if args.epochs > MAX_EPOCHS:
        print(f"\n[RULE ENFORCEMENT] Requested {args.epochs} epochs exceeds maximum allowed ({MAX_EPOCHS}).")
        print(f"Capping epochs to {MAX_EPOCHS}.\n")
        args.epochs = MAX_EPOCHS

    return args


def create_data_generators():
    """
    Creates train generator with real-time data augmentation and test generator with rescaling only.
    """
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print(f"ERROR: Training directory '{TRAIN_DIR}' or testing directory '{TEST_DIR}' not found.")
        print("Please run 'python dataset_info.py' first to verify/download the dataset.")
        sys.exit(1)

    print("[INFO] Setting up Data Generators...")

    # Train generator: Rescale + Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Test generator: Rescale only (never augment test data)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS,
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS,
        shuffle=False,
    )

    return train_generator, test_generator


def calculate_class_weights(train_generator):
    """
    Computes balanced class weights to compensate for class frequency skew (e.g. disgust having fewer samples).
    """
    classes = train_generator.classes
    unique_classes = np.unique(classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=classes,
    )
    class_weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}

    print("\n[INFO] Computed Balanced Class Weights:")
    for cls_idx, emotion in enumerate(EMOTIONS):
        w = class_weight_dict.get(cls_idx, 1.0)
        print(f"  - {emotion:<10}: {w:.4f}")
    return class_weight_dict


def train():
    args = parse_arguments()
    epochs = args.epochs

    train_gen, test_gen = create_data_generators()
    class_weights = calculate_class_weights(train_gen)

    print("\n[INFO] Building CNN Architecture...")
    model = build_emotion_model()
    save_model_summary(model)

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    lr_reducer = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    callbacks = [checkpoint, lr_reducer]

    print("\n" + "=" * 60)
    print(f"STARTING CNN TRAINING ({epochs} EPOCHS)")
    print("=" * 60)

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save history to outputs/history.json
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=4)
    print(f"\n[OK] Training history saved to: {HISTORY_PATH}")

    # Ensure best model is saved; if checkpoint did not trigger (rare single epoch), save current model
    if not os.path.exists(MODEL_PATH):
        model.save(MODEL_PATH)
        print(f"[OK] Model saved to: {MODEL_PATH}")
    else:
        print(f"[OK] Best model successfully checkpointed at: {MODEL_PATH}")


if __name__ == "__main__":
    train()
