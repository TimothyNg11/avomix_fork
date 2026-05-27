import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
import os
import sys
from collections import deque, Counter

# Compute the directory this script is in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "gesture_model.pth")
LABELS_FILE = os.path.join(BASE_DIR, "gesture_labels.json")

CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 5  # majority-vote over last N predictions


class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def normalize_landmarks(landmarks, width, height):
    """Same normalization used during data collection."""
    points = np.array([[lm.x * width, lm.y * height, lm.z * width]
                       for lm in landmarks])
    points -= points[0]
    scale = np.linalg.norm(points[9])
    if scale > 0:
        points /= scale
    return points.flatten()


def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        print(f"Error: {MODEL_FILE} or {LABELS_FILE} not found. Run train_model.py first.")
        sys.exit(1)

    with open(LABELS_FILE) as f:
        labels = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)

    model = GestureNet(checkpoint["input_size"], checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded model ({checkpoint['num_classes']} classes: {labels})")
    return model, labels, device


def main():
    model, labels, device = load_model()
    history = deque(maxlen=SMOOTHING_WINDOW)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_options = mp.tasks.BaseOptions(model_asset_path='./hand_landmarker.task')
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    print("\nRunning gesture recognition — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        gesture_text = "No hand"
        confidence = 0.0
        color = (100, 100, 100)

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            features = normalize_landmarks(lms, w, h)

            tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = probs.max(1)
                confidence = conf.item()
                predicted = labels[pred_idx.item()]

            # Smoothing: majority vote over recent frames
            history.append(predicted)
            if len(history) == SMOOTHING_WINDOW:
                most_common = Counter(history).most_common(1)[0][0]
                gesture_text = most_common
            else:
                gesture_text = predicted

            if confidence < CONFIDENCE_THRESHOLD:
                gesture_text = f"? ({gesture_text})"
                color = (0, 165, 255)  # orange
            else:
                color = (0, 255, 0)

            # Draw landmarks
            for lm in lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
        else:
            history.clear()

        # HUD
        cv2.putText(frame, f"Gesture: {gesture_text}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.0%}",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 2)
        cv2.putText(frame, "Q: quit",
                    (20, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 1)

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
