"""
app.py - Streamlit Interactive Web Application for Face Emotion Recognition
College Project: Face Emotion Recognition Using CNN
Subject: Advanced Machine Learning and Deep Learning

Tabs:
1. Live Webcam (Snapshot capture mode using st.camera_input)
2. Upload Image (Detects face, bounding box, probability bar chart)
3. Model Results (Accuracy metrics, training history, confusion matrix, class accuracy)
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from config import (
    MODEL_PATH,
    OUTPUTS_DIR,
    RESULTS_SUMMARY_PATH,
    CLASSIFICATION_REPORT_PATH,
    IMG_SIZE,
    EMOTIONS,
)

# Page configuration
st.set_page_config(
    page_title="Face Emotion Recognition - CNN",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    """
    Loads and caches the trained Keras CNN model.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def predict_emotion_from_face(gray_roi, model):
    """
    Preprocesses cropped face ROI and predicts emotion probabilities.
    """
    roi_resized = cv2.resize(gray_roi, IMG_SIZE, interpolation=cv2.INTER_AREA)
    roi_norm = roi_resized.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_norm, axis=(0, -1))
    preds = model.predict(roi_input, verbose=0)[0]
    return preds


def process_image(image_pil, model, face_cascade):
    """
    Detects faces, draws bounding boxes and emotion labels, returns annotated image and predictions.
    """
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(35, 35),
    )

    results = []
    annotated_bgr = img_bgr.copy()

    for idx, (x, y, w, h) in enumerate(faces):
        roi_gray = img_gray[y : y + h, x : x + w]
        preds = predict_emotion_from_face(roi_gray, model)
        top_idx = int(np.argmax(preds))
        top_emotion = EMOTIONS[top_idx]
        confidence = float(preds[top_idx] * 100.0)

        # Draw green bounding box
        cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{top_emotion.capitalize()} ({confidence:.1f}%)"
        cv2.rectangle(annotated_bgr, (x, max(0, y - 24)), (x + len(label) * 11, y), (0, 255, 0), -1)
        cv2.putText(
            annotated_bgr,
            label,
            (x + 4, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )

        results.append({
            "face_id": idx + 1,
            "box": (x, y, w, h),
            "dominant_emotion": top_emotion,
            "confidence": confidence,
            "probabilities": preds,
        })

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results


# Sidebar Navigation & Information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/facial-recognition-scan.png", width=70)
    st.title("Project Settings")
    st.markdown("---")

    # Model Status Check
    model_loaded = os.path.exists(MODEL_PATH)
    if model_loaded:
        st.success("✅ Model File: Found (`outputs/emotion_model.keras`)")
    else:
        st.error("❌ Model File: Missing (`outputs/emotion_model.keras`)")

    st.markdown("### 📚 Project Metadata")
    st.markdown("**Course:** Advanced ML & Deep Learning")
    st.markdown("**Dataset:** FER2013 (48x48 Grayscale)")
    st.markdown("**Classes:** 7 Facial Emotions")
    st.markdown(
        "- Angry, Disgust, Fear\n"
        "- Happy, Neutral, Sad, Surprise"
    )
    st.markdown("**Architecture:** 4-Block CNN + Dropout")
    st.markdown("---")
    st.caption("College Capstone Project &bull; Built with TensorFlow & Streamlit")

# Main Header
st.markdown('<p class="main-header">Face Emotion Recognition Using CNN</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Real-time facial expression analysis powered by Deep Learning (Convolutional Neural Networks) and OpenCV.</p>',
    unsafe_allow_html=True,
)

model = get_model()
face_cascade = get_face_cascade()

# Top level tabs
tab1, tab2, tab3 = st.tabs(["📷 Live Webcam", "🖼️ Upload Image", "📊 Model Results & Metrics"])

# =====================================================
# TAB 1: LIVE WEBCAM (Camera Snapshot)
# =====================================================
with tab1:
    st.subheader("Live Webcam Emotion Detection")
    st.write("Use your device's webcam to capture a selfie snapshot. The CNN will detect your face and analyze expression probabilities.")

    if model is None:
        st.warning("⚠️ Trained model not found! Please run `python train.py` in your terminal first to train the CNN.")
    else:
        cam_photo = st.camera_input("Take a photo with your webcam")
        if cam_photo is not None:
            input_pil = Image.open(cam_photo)
            annotated_img, results = process_image(input_pil, model, face_cascade)

            col_img, col_metrics = st.columns([3, 2])
            with col_img:
                st.image(annotated_img, caption="Analyzed Webcam Image", width=500)

            with col_metrics:
                if len(results) == 0:
                    st.warning("No face detected in the frame. Please align your face clearly with good lighting.")
                else:
                    for res in results:
                        st.success(f"**Detected Face #{res['face_id']}: {res['dominant_emotion'].capitalize()} ({res['confidence']:.1f}%)**")
                        prob_df = pd.DataFrame({
                            "Emotion": [e.capitalize() for e in EMOTIONS],
                            "Probability (%)": [p * 100.0 for p in res["probabilities"]],
                        }).set_index("Emotion")
                        st.bar_chart(prob_df, height=220)

# =====================================================
# TAB 2: UPLOAD IMAGE
# =====================================================
with tab2:
    st.subheader("Upload an Image for Analysis")
    st.write("Upload a portrait or selfie (JPG or PNG) to detect facial emotions.")

    if model is None:
        st.warning("⚠️ Trained model not found! Please run `python train.py` to create the model file.")
    else:
        uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file)
            annotated_img, results = process_image(image_pil, model, face_cascade)

            col1, col2 = st.columns([3, 2])
            with col1:
                st.image(annotated_img, caption="Predicted Emotions", width=500)

            with col2:
                if len(results) == 0:
                    st.warning("No face detected in this image. Try another photo with an upright, clear face.")
                else:
                    for res in results:
                        st.markdown(f"#### Face #{res['face_id']}: **{res['dominant_emotion'].capitalize()}**")
                        st.info(f"Confidence: **{res['confidence']:.2f}%**")
                        chart_data = pd.DataFrame({
                            "Emotion": [e.capitalize() for e in EMOTIONS],
                            "Confidence (%)": [p * 100.0 for p in res["probabilities"]],
                        }).set_index("Emotion")
                        st.bar_chart(chart_data, height=250)

# =====================================================
# TAB 3: MODEL RESULTS & PERFORMANCE METRICS
# =====================================================
with tab3:
    st.subheader("Model Evaluation & Training Metrics")
    st.write("View the quantitative performance of the 4-block CNN trained on the FER2013 benchmark dataset.")

    results_file = RESULTS_SUMMARY_PATH
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        # Top metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Test Accuracy", f"{summary.get('overall_test_accuracy_pct', 0.0)}%")
        m2.metric("Overall Test Loss", f"{summary.get('overall_test_loss', 0.0)}")
        m3.metric("Macro F1-Score", f"{summary.get('macro_f1', 0.0)}")
        m4.metric("Weighted F1-Score", f"{summary.get('weighted_f1', 0.0)}")
    else:
        st.info("ℹ️ Evaluation summary file (`outputs/results_summary.json`) not yet generated. Run `python evaluate.py` to produce it.")

    st.markdown("---")

    # Display Generated Diagnostic Charts
    hist_img = os.path.join(OUTPUTS_DIR, "training_history.png")
    cm_img = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    acc_img = os.path.join(OUTPUTS_DIR, "class_accuracy.png")
    dist_img = os.path.join(OUTPUTS_DIR, "class_distribution.png")
    sample_img = os.path.join(OUTPUTS_DIR, "sample_images.png")

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(hist_img):
            st.image(hist_img, caption="Training vs Validation (Accuracy & Loss)")
        else:
            st.warning("Training history graph missing. Run `python evaluate.py`.")

        if os.path.exists(dist_img):
            st.image(dist_img, caption="Dataset Class Distribution")

    with c2:
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix Heatmap (%)")
        else:
            st.warning("Confusion matrix graph missing. Run `python evaluate.py`.")

        if os.path.exists(acc_img):
            st.image(acc_img, caption="Per-Class Accuracy Breakdown")

    if os.path.exists(sample_img):
        st.markdown("---")
        st.image(sample_img, caption="Sample Faces per Emotion Class (48x48)")

    # Classification Report
    if os.path.exists(CLASSIFICATION_REPORT_PATH):
        st.markdown("---")
        st.subheader("Classification Report (Precision, Recall, F1-Score)")
        with open(CLASSIFICATION_REPORT_PATH, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
