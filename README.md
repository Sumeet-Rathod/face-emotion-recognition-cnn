# Face Emotion Recognition Using CNN

**Subject:** Advanced Machine Learning and Deep Learning  
**Project Type:** College Capstone Deep Learning Project  
**Frameworks:** TensorFlow / Keras, OpenCV, Streamlit, Scikit-Learn  
**Dataset:** FER2013 (48x48 Grayscale, 7 Emotion Classes)

---

## 📌 Project Overview
This project implements an end-to-end Facial Emotion Recognition (FER) system powered by a Deep Convolutional Neural Network (CNN). The model classifies facial expressions into seven emotion categories: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**.

The project is structured with modular Python scripts for data loading, neural network modeling, training with class balancing and data augmentation, comprehensive evaluation with diagnostic plots, a real-time OpenCV webcam demo with 5-frame temporal smoothing, and an interactive Streamlit web dashboard.

---

## 🏗️ CNN Model Architecture
The architecture is designed specifically for 48x48 single-channel grayscale facial imagery:

- **Input Layer:** 48 × 48 × 1
- **Convolutional Block 1:** `Conv2D(32, 3x3, ReLU, same padding)` &rarr; `BatchNormalization` &rarr; `MaxPooling2D(2x2)` &rarr; `Dropout(0.25)`
- **Convolutional Block 2:** `Conv2D(64, 3x3, ReLU, same padding)` &rarr; `BatchNormalization` &rarr; `MaxPooling2D(2x2)` &rarr; `Dropout(0.25)`
- **Convolutional Block 3:** `Conv2D(128, 3x3, ReLU, same padding)` &rarr; `BatchNormalization` &rarr; `MaxPooling2D(2x2)` &rarr; `Dropout(0.25)`
- **Convolutional Block 4:** `Conv2D(256, 3x3, ReLU, same padding)` &rarr; `BatchNormalization` &rarr; `MaxPooling2D(2x2)` &rarr; `Dropout(0.25)`
- **Dense Classification Head:**
  - `Flatten`
  - `Dense(512, ReLU)` &rarr; `BatchNormalization` &rarr; `Dropout(0.50)`
  - `Dense(256, ReLU)` &rarr; `Dropout(0.30)`
  - `Dense(7, Softmax)`
- **Optimizer & Loss:** Adam (`lr=0.001`), Categorical Cross-Entropy, Metric: Categorical Accuracy.

---

## 📂 Project Structure
```text
face_emotion_project/
│
├── config.py              # Configuration constants, hyper-parameters, paths
├── dataset_info.py        # Dataset verification, statistics, and distribution plots
├── model.py               # CNN architecture definition and summary export
├── train.py               # Training loop with augmentation, class weighting & callbacks
├── evaluate.py            # Quantitative metrics, confusion matrix, accuracy curves
├── realtime_demo.py       # Live OpenCV webcam emotion detection with side HUD
├── app.py                 # Interactive Streamlit multi-tab web application
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation and execution guide
│
├── data/                  # Dataset directory
│   ├── train/             # 7 emotion subfolders with training images
│   └── test/              # 7 emotion subfolders with testing images
│
└── outputs/               # Generated artifacts
    ├── emotion_model.keras        # Best saved model checkpoint
    ├── model_summary.txt          # Exported CNN layer summary
    ├── history.json               # Training history loss & accuracy per epoch
    ├── training_history.png       # Loss & accuracy progression plots
    ├── confusion_matrix.png       # 7x7 normalized confusion matrix heatmap
    ├── class_accuracy.png         # Per-class accuracy comparison bar chart
    ├── class_distribution.png     # Train vs test class distribution chart
    ├── sample_images.png          # Representative 48x48 sample faces grid
    ├── classification_report.txt  # Precision, recall, and F1-score breakdown
    ├── results_summary.json       # Top-level performance metrics summary
    └── screenshots/               # Saved frames from the real-time webcam demo
```

---

## 🚀 Execution Guide (Run Order)

Follow these exact steps to set up and run the project:

### 1. Setup Environment and Dependencies
Open your terminal in `face_emotion_project`:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (cmd.exe):
.\venv\Scripts\activate.bat
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Dataset and Generate Distribution Visualizations
```bash
python dataset_info.py
```
*Note: If the dataset is not yet in `data/`, the script will automatically attempt to fetch it via `kagglehub`. If download fails, it will provide exact instructions for manual extraction.*

### 3. Train the CNN Model
```bash
python train.py --epochs 5
```
*Note: The project enforces a strict cap of maximum 5 epochs to run efficiently on standard laptop CPUs and GPUs.*

### 4. Evaluate the Model on Test Data
```bash
python evaluate.py
```
*Loads the saved `emotion_model.keras` checkpoint and generates all charts and metric files in `outputs/`.*

### 5. Launch Real-Time Webcam Demo
```bash
python realtime_demo.py
```
*Controls:*
- `s`: Save an annotated screenshot to `outputs/screenshots/`
- `q`: Quit the webcam feed

### 6. Run the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
*Features 3 tabs: Live Webcam Snapshot, Image Upload Analysis, and Model Performance Dashboards.*

---

## 📊 Key Highlights
- **Class Imbalance Mitigation:** Incorporates scikit-learn balanced class weighting during training to handle skewed class representations (such as `disgust`).
- **Data Augmentation:** Real-time geometric augmentations (random rotations, width/height shifts, zoom, horizontal flips) applied exclusively to the training pipeline.
- **Temporal Prediction Smoothing:** A rolling 5-frame probability buffer stabilizes real-time webcam predictions and prevents flickering.
- **Lightweight & Portable:** Optimized for normal laptops without requiring dedicated GPU hardware.
