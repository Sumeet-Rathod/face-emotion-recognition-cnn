"""
dataset_info.py - Dataset Verification, Statistics, and Visualization
College Project: Face Emotion Recognition Using CNN
Subject: Advanced Machine Learning and Deep Learning

Verifies the FER2013 dataset structure (data/train and data/test),
downloads via kagglehub if missing, computes class distributions,
and generates class_distribution.png and sample_images.png.
"""

import os
import sys
import shutil
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR, TRAIN_DIR, TEST_DIR, OUTPUTS_DIR, EMOTIONS


def ensure_dataset():
    """
    Checks if data/train and data/test exist and are populated.
    If missing, attempts to download using kagglehub.
    If download fails, stops and instructs the user.
    """
    has_train = os.path.exists(TRAIN_DIR) and any(os.scandir(TRAIN_DIR))
    has_test = os.path.exists(TEST_DIR) and any(os.scandir(TEST_DIR))

    if has_train and has_test:
        print("[OK] Dataset directory exists at:", DATA_DIR)
        return True

    print("[INFO] Dataset not found in 'data/' folder. Attempting automatic download via kagglehub...")
    try:
        import kagglehub
        download_path = kagglehub.dataset_download("msambare/fer2013")
        print(f"[INFO] Downloaded FER2013 to: {download_path}")

        # Locate train and test inside downloaded path
        candidate_train = os.path.join(download_path, "train")
        candidate_test = os.path.join(download_path, "test")

        if not os.path.exists(candidate_train):
            # Check nested folders
            for root, dirs, _ in os.walk(download_path):
                if "train" in dirs and "test" in dirs:
                    candidate_train = os.path.join(root, "train")
                    candidate_test = os.path.join(root, "test")
                    break

        if os.path.exists(candidate_train) and os.path.exists(candidate_test):
            os.makedirs(DATA_DIR, exist_ok=True)
            print(f"[INFO] Copying train data to {TRAIN_DIR}...")
            shutil.copytree(candidate_train, TRAIN_DIR, dirs_exist_ok=True)
            print(f"[INFO] Copying test data to {TEST_DIR}...")
            shutil.copytree(candidate_test, TEST_DIR, dirs_exist_ok=True)
            print("[OK] FER2013 dataset successfully organized in data/train and data/test.")
            return True
        else:
            raise FileNotFoundError("Could not find 'train' and 'test' subdirectories in downloaded data.")

    except Exception as e:
        print("\n" + "!" * 70)
        print("ERROR: Automatic dataset download failed:", str(e))
        print("!" * 70)
        print("\nPLEASE MANUALLY PLACE THE DATASET AS FOLLOWS:")
        print("1. Download FER2013 from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013")
        print("2. Extract the archive into:")
        print(f"   {DATA_DIR}")
        print("3. Ensure the folder structure is:")
        print(f"   {TRAIN_DIR}/<emotion_name>/*.png")
        print(f"   {TEST_DIR}/<emotion_name>/*.png")
        print("   where emotions are: " + ", ".join(EMOTIONS))
        print("!" * 70 + "\n")
        sys.exit(1)


def count_images_in_dir(split_dir):
    """
    Counts images for each emotion class in a given split directory.
    """
    counts = {}
    for emotion in EMOTIONS:
        emotion_path = os.path.join(split_dir, emotion)
        if os.path.exists(emotion_path):
            images = glob.glob(os.path.join(emotion_path, "*.*"))
            counts[emotion] = len(images)
        else:
            counts[emotion] = 0
    return counts


def get_image_resolution(split_dir):
    """
    Reads one sample image to verify dimensions and mode.
    """
    for emotion in EMOTIONS:
        emotion_path = os.path.join(split_dir, emotion)
        if os.path.exists(emotion_path):
            images = glob.glob(os.path.join(emotion_path, "*.*"))
            if images:
                with Image.open(images[0]) as img:
                    return img.size, img.mode
    return (48, 48), "Grayscale"


def plot_class_distribution(train_counts, test_counts):
    """
    Generates and saves a clean side-by-side bar chart of class distributions.
    """
    x = np.arange(len(EMOTIONS))
    width = 0.35

    train_vals = [train_counts[e] for e in EMOTIONS]
    test_vals = [test_counts[e] for e in EMOTIONS]

    plt.figure(figsize=(10, 5), dpi=150)
    plt.bar(x - width / 2, train_vals, width, label="Train Set", color="#2563EB", edgecolor="black")
    plt.bar(x + width / 2, test_vals, width, label="Test Set", color="#10B981", edgecolor="black")

    plt.xlabel("Emotion Class", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Images", fontsize=11, fontweight="bold")
    plt.title("FER2013 Dataset - Class Distribution (Train vs Test)", fontsize=13, fontweight="bold")
    plt.xticks(x, [e.capitalize() for e in EMOTIONS], fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_file = os.path.join(OUTPUTS_DIR, "class_distribution.png")
    plt.savefig(out_file)
    plt.close()
    print(f"[OK] Saved class distribution chart to: {out_file}")


def plot_sample_images(split_dir=TRAIN_DIR):
    """
    Saves a 7-image sample grid showing one representative sample per emotion.
    """
    fig, axes = plt.subplots(1, 7, figsize=(14, 2.5), dpi=150)

    for idx, emotion in enumerate(EMOTIONS):
        ax = axes[idx]
        emotion_path = os.path.join(split_dir, emotion)
        images = glob.glob(os.path.join(emotion_path, "*.*")) if os.path.exists(emotion_path) else []

        if images:
            img = Image.open(images[0])
            ax.imshow(img, cmap="gray")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")

        ax.set_title(emotion.capitalize(), fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle("FER2013 - Representative Sample Faces per Emotion (48x48)", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()

    out_file = os.path.join(OUTPUTS_DIR, "sample_images.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved sample images grid to: {out_file}")


def main():
    ensure_dataset()

    train_counts = count_images_in_dir(TRAIN_DIR)
    test_counts = count_images_in_dir(TEST_DIR)

    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    total_images = total_train + total_test

    img_size, img_mode = get_image_resolution(TRAIN_DIR)

    print("\n" + "=" * 60)
    print("FER2013 DATASET INFORMATION")
    print("=" * 60)
    print(f"Image Resolution : {img_size[0]} x {img_size[1]} pixels")
    print(f"Color Mode       : {img_mode}")
    print(f"Total Images     : {total_images} (Train: {total_train}, Test: {total_test})")
    print("-" * 60)
    print(f"{'Emotion':<12} | {'Train Count':<12} | {'Test Count':<12} | {'Total':<10}")
    print("-" * 60)
    for emotion in EMOTIONS:
        tr = train_counts[emotion]
        te = test_counts[emotion]
        print(f"{emotion.capitalize():<12} | {tr:<12} | {te:<12} | {tr + te:<10}")
    print("=" * 60)

    plot_class_distribution(train_counts, test_counts)
    plot_sample_images(TRAIN_DIR)


if __name__ == "__main__":
    main()
