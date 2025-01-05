"""
Niko Tsiolas 
January 2025

Emotion classification using MediaPipe Holistic for face landmarks
and a RandomForest classifier for inference. 

Classes (emotions): anger, disgust, fear, happiness, sadness, surprise
Frames per sequence: 30
Sequences per class: 5 (default)

STILL WORK IN PROGRESS


Usage:
1. (Optional) Collect data by uncommenting collect_data() in the main section.
2. Load data, train model, and evaluate.
3. (Optional) Run real-time detection.
"""

import os
import time
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
actions = np.array(['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])
no_sequences = 5
sequence_length = 30
DATA_PATH = "MP_Face_Data_Sklearn"
os.makedirs(DATA_PATH, exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def process_frame(image, model):
    """Convert BGR image to RGB, process via MediaPipe, return results + annotated BGR image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = model.process(rgb)
    rgb.flags.writeable = True
    annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return annotated, results

def draw_facemesh(image, results):
    """
    Draw face landmarks using FACEMESH_TESSELATION. 
    (Holistic does not have FACE_CONNECTIONS).
    """
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1, circle_radius=1),
        )

def extract_keypoints(results):
    """
    Extract face landmarks (468 * 3 = 1404 coords).
    Return zero array if no face is detected.
    """
    face_data = np.zeros(468 * 3, dtype=np.float32)
    if results.face_landmarks:
        coords = []
        for lm in results.face_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        face_data = np.array(coords, dtype=np.float32)
    return face_data


# ------------------------------------------------------------------
# Data Collection
# ------------------------------------------------------------------
def collect_data():
    """
    Use webcam to record sequences of face landmarks for each emotion.
    Each sequence has 'sequence_length' frames, and 'no_sequences' total sequences per emotion.
    """
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            action_dir = os.path.join(DATA_PATH, action)
            os.makedirs(action_dir, exist_ok=True)

            existing = os.listdir(action_dir)
            dirmax = -1
            if existing:
                dirmax = np.max(np.array(existing).astype(int))

            print(f"\nCollecting data for {action}")
            time.sleep(1)

            for seq_idx in range(no_sequences):
                seq_path = os.path.join(action_dir, str(dirmax + 1 + seq_idx))
                os.makedirs(seq_path, exist_ok=True)
                print(f"Sequence {seq_idx} for '{action}'")

                for frame_idx in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    annotated, results = process_frame(frame, holistic)
                    draw_facemesh(annotated, results)

                    cv2.putText(
                        annotated,
                        f"{action} | Seq: {seq_idx} | Frame: {frame_idx}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                    cv2.imshow("Data Collection", annotated)

                    keypoints = extract_keypoints(results)
                    np.save(os.path.join(seq_path, f"{frame_idx}.npy"), keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection finished.")


# ------------------------------------------------------------------
# Dataset Assembly
# ------------------------------------------------------------------
def load_dataset():
    """
    Traverse saved .npy files for each emotion, build feature (X) and label (y) arrays.
    Each 30-frame sequence is flattened into a 1D vector of length (30 * 1404) = 42120.
    """
    sequences, labels = [], []
    label_map = {label: i for i, label in enumerate(actions)}

    for action in actions:
        path = os.path.join(DATA_PATH, action)
        if not os.path.isdir(path):
            continue

        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            frames_data = []
            for fidx in range(sequence_length):
                fpath = os.path.join(folder_path, f"{fidx}.npy")
                if not os.path.exists(fpath):
                    frames_data = []
                    break
                arr = np.load(fpath)
                frames_data.append(arr)

            if len(frames_data) == sequence_length:
                sequences.append(np.array(frames_data).flatten())
                labels.append(label_map[action])

    X = np.array(sequences)
    y = np.array(labels)
    print(f"Loaded dataset:\nX shape: {X.shape}\ny shape: {y.shape}")
    return X, y


# ------------------------------------------------------------------
# Training & Inference
# ------------------------------------------------------------------
def train_classifier(X, y):
    """
    Train a simple RandomForest model on the collected face landmarks.
    """
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

def real_time_detection(model):
    """
    Run real-time face landmark detection + classification via webcam.
    Keeps a rolling buffer of 30 frames and predicts once it's full.
    """
    buffer = []
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, results = process_frame(frame, holistic)
            draw_facemesh(annotated, results)

            keypoints = extract_keypoints(results)
            buffer.append(keypoints)
            buffer = buffer[-sequence_length:]  # keep last 30 frames

            if len(buffer) == sequence_length:
                pred = model.predict(np.array(buffer).flatten().reshape(1, -1))[0]
                emotion = actions[pred]

                cv2.putText(
                    annotated,
                    f"Prediction: {emotion}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow("Real-Time Detection", annotated)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Uncomment to collect new data:
    collect_data()

    X, y = load_dataset()
    if len(y) == 0:
        print("No data found. Please collect data first.")
        exit(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    model = train_classifier(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=actions))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Uncomment to run real-time detection:
    real_time_detection(model)
