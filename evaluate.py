"""
evaluate.py - Quantitative & Qualitative Performance Evaluation
College Project: Face Emotion Recognition Using CNN
Subject: Advanced Machine Learning and Deep Learning

Evaluates the trained CNN model on the test set and generates:
- outputs/training_history.png (Accuracy and Loss curves)
- outputs/confusion_matrix.png (Normalized percentage heatmap across all 7 emotions)
- outputs/class_accuracy.png (Per-class accuracy bar chart)
- outputs/classification_report.txt (Precision, recall, F1 per class)
- outputs/results_summary.json (Overall accuracy, loss, macro F1)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import (
    TEST_DIR,
    MODEL_PATH,
    HISTORY_PATH,
    OUTPUTS_DIR,
    RESULTS_SUMMARY_PATH,
    CLASSIFICATION_REPORT_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    EMOTIONS,
)


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. Please run 'python train.py' first."
        )
    print(f"[INFO] Loading trained model from: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def load_test_generator():
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory '{TEST_DIR}' does not exist.")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS,
        shuffle=False,
    )
    return test_generator


def plot_training_history():
    """
    Plots training and validation accuracy & loss curves from outputs/history.json.
    """
    if not os.path.exists(HISTORY_PATH):
        print(f"[WARNING] History file not found at '{HISTORY_PATH}'. Skipping history plot.")
        return

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Accuracy Plot
    axes[0].plot(epochs, history["accuracy"], "o-", color="#2563EB", label="Training Accuracy", linewidth=2)
    axes[0].plot(epochs, history["val_accuracy"], "s--", color="#10B981", label="Validation Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy Across Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Accuracy", fontsize=10)
    axes[0].set_xticks(list(epochs))
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    axes[1].plot(epochs, history["loss"], "o-", color="#EF4444", label="Training Loss", linewidth=2)
    axes[1].plot(epochs, history["val_loss"], "s--", color="#F59E0B", label="Validation Loss", linewidth=2)
    axes[1].set_title("Model Loss Across Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Categorical Cross-Entropy Loss", fontsize=10)
    axes[1].set_xticks(list(epochs))
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Face Emotion CNN - Training & Validation Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = os.path.join(OUTPUTS_DIR, "training_history.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved training curves to: {out_file}")


def plot_confusion_matrix(y_true, y_pred):
    """
    Plots a normalized confusion matrix heatmap showing classification percentages.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true") * 100.0
    labels = [e.capitalize() for e in EMOTIONS]

    plt.figure(figsize=(9, 7), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Percentage (%)"},
    )
    plt.title("FER2013 - Normalized Confusion Matrix (%)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Emotion", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Emotion", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_file = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[OK] Saved confusion matrix heatmap to: {out_file}")
    return cm


def plot_class_accuracy(cm):
    """
    Plots a bar chart showing individual classification accuracy for each emotion.
    """
    class_accuracies = np.diag(cm)
    labels = [e.capitalize() for e in EMOTIONS]

    plt.figure(figsize=(10, 5), dpi=150)
    bars = plt.bar(labels, class_accuracies, color="#3B82F6", edgecolor="black", width=0.55)

    # Annotate percentage on top of each bar
    for bar, acc in zip(bars, class_accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title("FER2013 - Per-Class Test Accuracy", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Emotion Class", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_file = os.path.join(OUTPUTS_DIR, "class_accuracy.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[OK] Saved per-class accuracy bar chart to: {out_file}")

    return {e: float(acc) for e, acc in zip(EMOTIONS, class_accuracies)}


def evaluate():
    model = load_trained_model()
    test_gen = load_test_generator()

    print("\n[INFO] Evaluating model on test set...")
    eval_results = model.evaluate(test_gen, verbose=1)
    test_loss = float(eval_results[0])
    test_accuracy = float(eval_results[1])

    print("[INFO] Computing predictions for confusion matrix & classification report...")
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    labels = [e.capitalize() for e in EMOTIONS]

    # Classification Report
    report_str = classification_report(y_true, y_pred, target_names=labels, digits=4)
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Face Emotion Recognition - Test Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report_str + "\n")
    print(f"[OK] Saved classification report to: {CLASSIFICATION_REPORT_PATH}")

    # Macro F1
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    # Visualizations
    plot_training_history()
    cm = plot_confusion_matrix(y_true, y_pred)
    per_class_acc = plot_class_accuracy(cm)

    # Save summary json
    summary_data = {
        "overall_test_accuracy": test_accuracy,
        "overall_test_accuracy_pct": round(test_accuracy * 100.0, 2),
        "overall_test_loss": round(test_loss, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_accuracy_pct": per_class_acc,
    }

    with open(RESULTS_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[OK] Saved results summary to: {RESULTS_SUMMARY_PATH}")

    # Print clear final metrics in terminal
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"Overall Test Loss     : {test_loss:.4f}")
    print(f"Macro F1-Score        : {macro_f1:.4f}")
    print(f"Weighted F1-Score     : {weighted_f1:.4f}")
    print("-" * 60)
    print("Per-Class Accuracy:")
    for emotion, acc in per_class_acc.items():
        print(f"  - {emotion.capitalize():<10}: {acc:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate()
