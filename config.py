import os

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SCREENSHOTS_DIR = os.path.join(OUTPUTS_DIR, "screenshots")

# File paths for models, metrics, and summaries
MODEL_PATH = os.path.join(OUTPUTS_DIR, "emotion_model.keras")
HISTORY_PATH = os.path.join(OUTPUTS_DIR, "history.json")
MODEL_SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "model_summary.txt")
RESULTS_SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "results_summary.json")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUTS_DIR, "classification_report.txt")

# Image and training hyper-parameters
IMG_HEIGHT = 48
IMG_WIDTH = 48
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 1  # Grayscale
BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
MAX_EPOCHS = 5
LEARNING_RATE = 0.001

# FER2013 7 Emotion categories in standard alphabetical order
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# Ensure required directories exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
