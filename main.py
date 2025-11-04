import numpy as np
import os
import mediapipe as mp
import cv2
import keyboard
import torch
import torch.nn as nn
import torch.nn.functional as F  # --- ADDED ---
from PIL import ImageFont, ImageDraw, Image
from llm_sentence_generator import LLMSentenceGenerator

# import joblib  # --- REMOVED ---

# -------------------------------
# CONFIG
# -------------------------------
SEQUENCE_LENGTH = 20  # Match training sequence length
CONFIDENCE_THRESHOLD = 0.9
FONT_PATH = "font.ttf"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def image_process(image, model):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# Import the raw keypoint extraction function (extracts 225 features)
from my_functions import keypoint_extraction


def draw_armenian_text(img, text, position, font_path=FONT_PATH, font_size=32, color=(255, 255, 255)):
    if not text:
        return img
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# -----------------------------------------------
# --- MODIFIED: ADDED PREPROCESSING FROM TRAIN SCRIPT ---
# -----------------------------------------------
def preprocess_keypoints(sequence):
    """
    Applies normalization and motion deltas to the raw keypoint sequence.
    Input shape: (20, 225)
    Output shape: (20, 450)
    """
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(sequence)

    # --- A. Normalization (Translation Invariance) ---
    # Reshape to (seq_len, num_groups, num_keypoints, num_coords)
    pose = sequence[:, :99].reshape(SEQUENCE_LENGTH, 33, 3)
    lh = sequence[:, 99:162].reshape(SEQUENCE_LENGTH, 21, 3)
    rh = sequence[:, 162:225].reshape(SEQUENCE_LENGTH, 21, 3)

    # Get reference points (wrist for hands, nose for pose)
    nose = pose[:, 0:1, :]
    lwrist = lh[:, 0:1, :]
    rwrist = rh[:, 0:1, :]

    # Subtract the reference point.
    pose_norm = pose - nose
    lh_norm = lh - lwrist
    rh_norm = rh - rwrist

    # Flatten back to (20, 225)
    normalized_sequence = np.concatenate([
        pose_norm.reshape(SEQUENCE_LENGTH, 99),
        lh_norm.reshape(SEQUENCE_LENGTH, 63),
        rh_norm.reshape(SEQUENCE_LENGTH, 63)
    ], axis=1)

    # --- B. Motion Deltas (Velocity) ---
    deltas = np.diff(normalized_sequence, axis=0)
    deltas = np.concatenate([np.zeros((1, 225)), deltas], axis=0)  # Pad first frame

    # --- C. Concatenate Features ---
    # Final shape: (20, 450)
    final_features = np.concatenate([normalized_sequence, deltas], axis=1)

    return final_features


# -----------------------------------------------
# --- MODIFIED: MODEL DEFINITION (same as training) ---
# -----------------------------------------------
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
            input_size,
            hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
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


# -------------------------------
# MAIN APPLICATION
# -------------------------------

try:
    # --- MODIFIED: Load class_mapping.npy (which is a dict) ---
    class_to_idx = np.load('class_mapping.npy', allow_pickle=True).item()

    # --- MODIFIED: Recreate the actions list in the correct order ---
    num_actions = len(class_to_idx)
    actions = [""] * num_actions
    for class_name, class_idx in class_to_idx.items():
        actions[class_idx] = class_name

    print(f"Loaded actions from class_mapping.npy: {actions}")

except Exception as e:
    print(f"Error loading class_mapping.npy: {e}")
    print("Train the model first (train_model.py) to generate class_mapping.npy")
    exit()

try:
    # --- MODIFIED: Load PyTorch GRU model from best_model.pth ---
    checkpoint_path = 'best_model.pth'  # This matches train_model.py
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} not found. Please train the model first.")

    # Create model with INPUT_SIZE = 450
    model = SignLanguageGRU(input_size=450, num_classes=len(actions)).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- FIXED: Use .load_state_dict() instead of .load() ---
    model.load_state_dict(checkpoint)
    print(f"Model loaded from {checkpoint_path}")

    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback

    traceback.print_exc()
    print("Make sure you have trained the model first by running train_model.py")
    exit()

# --- MODIFIED: REMOVED SCALER LOADING ---
print("Preprocessing (Normalization + Deltas) is built into this script.")
print("No feature_scaler.pkl needed.")

# Validate that model output size matches number of actions
num_actions = len(actions)
print(f"Model configured for {num_actions} classes")

sentence, keypoints, last_prediction = [], [], None
prediction_history = []  # Store recent predictions for smoothing
word_buffer = []  # Buffer to collect words before generating sentence
collecting_words = False  # Flag to indicate if we're collecting words

# Initialize LLM sentence generator
sentence_generator = LLMSentenceGenerator()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image, results = image_process(image, holistic)
        draw_landmarks(image, results)

        # Extract raw keypoints and validate shape
        current_keypoints = keypoint_extraction(results)
        if current_keypoints.shape[0] != 225:
            print(f"Warning: Expected 225 features, got {current_keypoints.shape[0]}")
            continue

        keypoints.append(current_keypoints)
        keypoints = keypoints[-SEQUENCE_LENGTH:]

        if len(keypoints) == SEQUENCE_LENGTH:
            keypoints_array = np.array(keypoints)  # Shape: (20, 225)

            # Validate the raw array shape
            if keypoints_array.shape != (SEQUENCE_LENGTH, 225):
                print(f"Warning: Expected shape ({SEQUENCE_LENGTH}, 225), got {keypoints_array.shape}")
                continue

            # --- MODIFIED: Apply the SAME preprocessing as training ---
            processed_keypoints = preprocess_keypoints(keypoints_array)  # Shape: (20, 450)

            # Convert to PyTorch tensor and predict
            keypoints_tensor = torch.FloatTensor(processed_keypoints).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(keypoints_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                max_prob, predicted_class = torch.max(probabilities, dim=1)
                max_prob = max_prob.item()
                predicted_class = predicted_class.item()

            action = actions[predicted_class]

            # (Rest of your prediction/smoothing logic is unchanged)

            # Add to prediction history for smoothing
            prediction_history.append((action, max_prob))
            if len(prediction_history) > 5:  # Keep only last 5 predictions
                prediction_history.pop(0)

            # Debug information
            print(f"Prediction: {action} (confidence: {max_prob:.3f})")

            # Only add prediction if confidence is high enough
            if max_prob > CONFIDENCE_THRESHOLD:
                # Check if this prediction appears consistently in recent history
                recent_actions = [pred[0] for pred in prediction_history[-3:]]  # Last 3 predictions
                if recent_actions.count(action) >= 2:  # Appears at least 2 times in last 3
                    if action != last_prediction:
                        if collecting_words:
                            word_buffer.append(action)
                            print(f"Added to word buffer: {action}")
                        else:
                            sentence.append(action)
                            last_prediction = action
                            print(f"Added to sentence: {action} (smoothed)")
                # else:
                # print(f"Prediction not consistent: {action} appears {recent_actions.count(action)}/3 times")
            else:
                print(f"Low confidence: {max_prob:.3f} < {CONFIDENCE_THRESHOLD}")

            if len(sentence) > 5:
                sentence = sentence[-5:]

        # (Rest of your keyboard/display logic is unchanged)

        if keyboard.is_pressed(' '):
            if not collecting_words:
                # First spacebar: Start collecting words
                collecting_words = True
                word_buffer = []
                last_prediction = None
                print("🎯 Started collecting words. Sign your words, then press SPACE again to generate sentence.")
            else:
                # Second spacebar: Generate sentence from collected words
                collecting_words = False
                if word_buffer:
                    # Generate sentence from collected words
                    meaningful_sentence = sentence_generator.generate_sentence(set(word_buffer))
                    sentence.append(meaningful_sentence)
                    print(f"📝 Generated sentence: {meaningful_sentence}")
                    print(f"📝 From words: {word_buffer}")
                else:
                    print("⚠️ No words collected")
                word_buffer = []
                last_prediction = None

        # Display current state
        if collecting_words:
            if word_buffer:
                display_text = f"Collecting: {' '.join(word_buffer)} (Press SPACE to generate sentence)"
            else:
                display_text = "Collecting words... (Press SPACE to generate sentence)"
            # print(f"📝 Word buffer: {word_buffer}")
        elif sentence:
            display_text = sentence[-1] if sentence else "No sentence yet"
            # print(f"📝 Current sentence: {display_text}")
        else:
            display_text = "Press SPACE to start collecting words"
            # print("Waiting for SPACE to start...")
        text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x_pos = (image.shape[1] - text_size[0]) // 2 if text_size[0] < image.shape[1] else 0
        y_pos = image.shape[0] - 40

        image = draw_armenian_text(image, display_text, (x_pos, y_pos), font_size=36, color=(255, 255, 0))

        cv2.imshow('Real-Time Sign Language Recognition', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
