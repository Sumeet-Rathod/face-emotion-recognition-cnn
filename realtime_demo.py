"""
realtime_demo.py - Live Webcam Face Emotion Recognition Demo
College Project: Face Emotion Recognition Using CNN
Subject: Advanced Machine Learning and Deep Learning

Features:
- Camera detection (tries index 0, then 1)
- Haar Cascade frontal face detection
- Preprocessing: crop ROI, resize to 48x48, normalize 0-1
- 5-frame rolling average prediction smoothing (flicker reduction)
- Green bounding box with predicted emotion & confidence percentage
- Side dashboard panel with dynamic probability bars for all 7 emotions
- Keyboard controls: 's' to take screenshot, 'q' to quit
"""

import os
import sys
import time
from collections import deque
import cv2
import numpy as np
import tensorflow as tf

from config import MODEL_PATH, SCREENSHOTS_DIR, IMG_SIZE, EMOTIONS


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("\n" + "=" * 60)
        print("ERROR: Trained model file not found!")
        print(f"Path: {MODEL_PATH}")
        print("Please run 'python train.py' first to train and save the model.")
        print("=" * 60 + "\n")
        sys.exit(1)

    print(f"[INFO] Loading emotion model from '{MODEL_PATH}'...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)


def open_camera():
    """
    Attempts to open webcam on index 0, then falls back to index 1.
    """
    for index in (0, 1):
        print(f"[INFO] Attempting to open camera on index {index}...")
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[OK] Camera successfully connected on index {index}.")
                return cap
            cap.release()

    print("\n" + "=" * 60)
    print("ERROR: Webcam not found on index 0 or index 1.")
    print("Please verify that your webcam is plugged in, permitted, and not in use by another app.")
    print("=" * 60 + "\n")
    return None


def draw_side_panel(panel, probabilities, dominant_idx):
    """
    Renders emotion probability bars and instructions on a 300px dark sidebar panel.
    """
    h, w, _ = panel.shape
    panel[:] = (26, 26, 26)  # Dark background #1a1a1a

    # Header
    cv2.putText(panel, "EMOTION METRICS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.line(panel, (20, 48), (w - 20, 48), (70, 70, 70), 1)

    bar_x = 100
    max_bar_width = w - bar_x - 65
    start_y = 80
    row_height = 42

    for i, emotion in enumerate(EMOTIONS):
        prob = probabilities[i]
        y = start_y + i * row_height

        # Highlight dominant emotion in green, others in light gray/blue
        is_dominant = (i == dominant_idx)
        text_color = (0, 255, 128) if is_dominant else (200, 200, 200)
        bar_color = (0, 220, 100) if is_dominant else (180, 120, 50)

        # Emotion Name
        cv2.putText(panel, emotion.capitalize()[:7], (20, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        # Background Bar
        cv2.rectangle(panel, (bar_x, y), (bar_x + max_bar_width, y + 16), (50, 50, 50), -1)

        # Filled Probability Bar
        fill_w = int(max_bar_width * prob)
        if fill_w > 0:
            cv2.rectangle(panel, (bar_x, y), (bar_x + fill_w, y + 16), bar_color, -1)

        # Percentage text
        cv2.putText(
            panel,
            f"{prob * 100:4.1f}%",
            (bar_x + max_bar_width + 8, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            text_color,
            1,
        )

    # Footer instructions
    cv2.line(panel, (20, h - 60), (w - 20, h - 60), (70, 70, 70), 1)
    cv2.putText(panel, "Press 's' : Save Screenshot", (20, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
    cv2.putText(panel, "Press 'q' : Quit Demo", (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)


def main():
    model = load_model()
    cap = open_camera()
    if cap is None:
        sys.exit(1)

    # Load OpenCV Haar Cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"ERROR: Could not load Haar cascade classifier from {cascade_path}")
        sys.exit(1)

    # Prediction smoothing deque over last 5 frames
    prediction_buffer = deque(maxlen=5)
    default_probabilities = np.zeros(len(EMOTIONS))
    dominant_index = 0

    print("\n[OK] Real-time emotion demo is running. Focus on OpenCV window.")
    print("     Controls: Press 's' to save screenshot | Press 'q' to exit.\n")

    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    screenshot_notice_time = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Empty frame received from webcam. Exiting...")
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40),
        )

        if len(faces) > 0:
            # Focus on largest face in the frame
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            fx, fy, fw, fh = largest_face

            # Extract ROI, resize to 48x48, normalize 0-1
            roi_gray = gray[fy : fy + fh, fx : fx + fw]
            try:
                roi_resized = cv2.resize(roi_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
                roi_normalized = roi_resized.astype("float32") / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=(0, -1))  # Shape: (1, 48, 48, 1)

                raw_preds = model.predict(roi_input, verbose=0)[0]
                prediction_buffer.append(raw_preds)
            except Exception as e:
                pass

            # Compute smoothed probabilities
            if len(prediction_buffer) > 0:
                current_probs = np.mean(prediction_buffer, axis=0)
                dominant_index = int(np.argmax(current_probs))
                dominant_emotion = EMOTIONS[dominant_index]
                conf = current_probs[dominant_index] * 100.0

                # Draw green bounding box & label
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                label = f"{dominant_emotion.capitalize()} ({conf:.1f}%)"
                cv2.rectangle(frame, (fx, fy - 26), (fx + len(label) * 11, fy), (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    label,
                    (fx + 4, fy - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                )
                default_probabilities = current_probs
        else:
            # Decay buffer if face is lost
            if len(prediction_buffer) > 0:
                prediction_buffer.popleft()

        # Side panel for metrics
        panel_width = 300
        side_panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
        draw_side_panel(side_panel, default_probabilities, dominant_index)

        # Combine webcam feed + side panel
        combined_display = np.hstack((frame, side_panel))

        # Show on-screen notification if a screenshot was taken recently
        if time.time() - screenshot_notice_time < 2.0:
            cv2.putText(
                combined_display,
                "Screenshot Saved!",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Face Emotion Recognition - CNN Demo", combined_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit key detected. Closing webcam...")
            break
        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            save_path = os.path.join(SCREENSHOTS_DIR, filename)
            cv2.imwrite(save_path, combined_display)
            screenshot_notice_time = time.time()
            print(f"[OK] Screenshot saved to: {save_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
