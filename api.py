import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import cv2
import uvicorn
import tempfile
from fastapi import FastAPI, UploadFile, File
from pathlib import Path

# --- CONFIG ---
# These must match the values used during training
SEQUENCE_LENGTH = 20
MODEL_PATH = "best_model.pth"
MAPPING_PATH = "class_mapping.npy"


# --- 1. MODEL DEFINITION (Copied from train_model.py) ---
# This must be identical to the model you trained

class Attention(nn.Module):
    """Simple dot-product attention."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weights = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        scores = torch.matmul(x, self.weights)
        attn_weights = F.softmax(scores, dim=1)
        context_vector = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return context_vector, attn_weights


class SignLanguageGRU(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_gru_layers=2, dropout=0.3):
        super(SignLanguageGRU, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_gru_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )
        gru_output_size = hidden_size * 2
        self.bn1 = nn.BatchNorm1d(gru_output_size)
        self.attention = Attention(gru_output_size)
        self.fc1 = nn.Linear(gru_output_size, 64)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        context_vector, _ = self.attention(gru_out)
        bn_out = self.bn1(context_vector)
        x = self.relu(self.fc1(bn_out))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


# --- 2. PREPROCESSING FUNCTIONS (Copied from scripts) ---

def extract_keypoints(results):
    """
    Extracts 225 raw keypoints from MediaPipe results.
    (Copied from data_augmentation.py)
    """
    pose = np.array([[lm.x, lm.y, lm.z] for lm in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, lh, rh])


def preprocess_keypoints(sequence, seq_len):
    """
    Applies normalization and motion deltas to the raw keypoint sequence.
    (Copied from preprocess_data.py)
    """
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(sequence)

    pose = sequence[:, :99].reshape(seq_len, 33, 3)
    lh = sequence[:, 99:162].reshape(seq_len, 21, 3)
    rh = sequence[:, 162:225].reshape(seq_len, 21, 3)

    nose = pose[:, 0:1, :]
    lwrist = lh[:, 0:1, :]
    rwrist = rh[:, 0:1, :]

    pose_norm = pose - nose
    lh_norm = lh - lwrist
    rh_norm = rh - rwrist

    normalized_sequence = np.concatenate([
        pose_norm.reshape(seq_len, 99),
        lh_norm.reshape(seq_len, 63),
        rh_norm.reshape(seq_len, 63)
    ], axis=1)

    deltas = np.diff(normalized_sequence, axis=0)
    deltas = np.concatenate([np.zeros((1, 225)), deltas], axis=0)

    final_features = np.concatenate([normalized_sequence, deltas], axis=1)
    return final_features


# --- 3. LOAD MODELS & DATA (Global) ---
# Load all models once when the API starts

try:
    print("Loading models and data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load class mapping
    class_mapping = np.load(MAPPING_PATH, allow_pickle=True).item()
    num_classes = len(class_mapping)
    # Create reverse mapping (index -> name)
    actions = {v: k for k, v in class_mapping.items()}
    print(f"Loaded {num_classes} classes.")

    # Load PyTorch model
    model = SignLanguageGRU(input_size=450, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Load MediaPipe model
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("MediaPipe model loaded.")

except FileNotFoundError as e:
    print(f"Error: Missing required file. {e}")
    print(f"Please make sure '{MODEL_PATH}' and '{MAPPING_PATH}' are in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred during initialization: {e}")
    exit()

# --- 4. INITIALIZE API ---
app = FastAPI()

print("API is ready to accept requests.")


# --- 5. DEFINE API ENDPOINT ---

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    """
    Accepts a video file, processes it, and returns the predicted sign.
    """

    # Save the uploaded file to a temporary file
    # cv2.VideoCapture needs a file path, not in-memory bytes
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            temp_video_path = temp_video.name

            # --- Process the video (same logic as data augmentation) ---
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                return {"error": "Video is empty or corrupted."}

            indices_to_sample = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
            keypoints_sequence = []

            for frame_idx in indices_to_sample:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = mp_holistic.process(image_rgb)

                keypoints = extract_keypoints(results)
                keypoints_sequence.append(keypoints)

            cap.release()

            # --- Pad sequence if video was too short ---
            if len(keypoints_sequence) == 0:
                return {"error": "Could not extract any keypoints from the video."}

            while len(keypoints_sequence) < SEQUENCE_LENGTH:
                keypoints_sequence.append(keypoints_sequence[-1])  # Pad with last frame

            # Ensure it's exactly SEQUENCE_LENGTH
            keypoints_array = np.array(keypoints_sequence[:SEQUENCE_LENGTH])

            if keypoints_array.shape != (SEQUENCE_LENGTH, 225):
                return {
                    "error": f"Keypoint array shape mismatch. Expected {(SEQUENCE_LENGTH, 225)}, got {keypoints_array.shape}"}

            # --- Preprocess and Predict ---
            processed_keypoints = preprocess_keypoints(keypoints_array, SEQUENCE_LENGTH)  # (20, 450)
            keypoints_tensor = torch.FloatTensor(processed_keypoints).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(keypoints_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                max_prob, predicted_class_idx = torch.max(probabilities, dim=1)

            predicted_sign = actions.get(predicted_class_idx.item(), "Unknown")
            confidence = max_prob.item()

            # Return the prediction
            return {
                "predicted_sign": predicted_sign,
                "confidence": confidence,
                "note": f"Video processed {len(indices_to_sample)} frames out of {total_frames}."
            }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


# --- 6. RUN THE API ---
if __name__ == "__main__":
    # This makes the script runnable with: python api.py
    # It will run on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)