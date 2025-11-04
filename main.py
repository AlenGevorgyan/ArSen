import numpy as np
import os
import mediapipe as mp
import cv2
import keyboard
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from llm_sentence_generator import LLMSentenceGenerator
import joblib

# -------------------------------
# CONFIG
# -------------------------------
SEQUENCE_LENGTH = 10  # Match training sequence length
CONFIDENCE_THRESHOLD = 0.9
FONT_PATH = "font.ttf"

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


# Import the corrected keypoint extraction function
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


# -------------------------------
# MAIN APPLICATION
# -------------------------------

try:
    actions = np.load('actions.npy')
    print(f"Loaded actions from actions.npy: {actions}")
except Exception as e:
    print(f"Error loading actions.npy: {e}")
    print("Train the model first (train_model.py) to generate actions.npy")
    exit()

try:
    model = load_model('my_model.h5')
    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    raise e

try:
    scaler = joblib.load('feature_scaler.pkl')
    print("Feature scaler loaded.")
except Exception as e:
    scaler = None
    print("Warning: feature_scaler.pkl not found. Proceeding without standardization.")
    print(f"Error loading model: {e}")
    print("Make sure you have trained the model first by running train_model.py")
    exit()

# Validate that model output size matches number of actions
num_actions = len(actions)
model_outputs = model.output_shape[-1]
if model_outputs != num_actions:
    print(f"Error: Model outputs {model_outputs} classes, but actions.npy has {num_actions} labels.")
    print("Ensure you're using the matching model and actions.npy from the same training run.")
    exit()

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

        # Extract keypoints and validate shape
        current_keypoints = keypoint_extraction(results)
        if current_keypoints.shape[0] != 225:
            print(f"Warning: Expected 225 features, got {current_keypoints.shape[0]}")
            continue
            
        keypoints.append(current_keypoints)
        keypoints = keypoints[-SEQUENCE_LENGTH:]

        if len(keypoints) == SEQUENCE_LENGTH:
            keypoints_array = np.array(keypoints)
            # Validate the array shape before prediction
            if keypoints_array.shape != (SEQUENCE_LENGTH, 225):
                print(f"Warning: Expected shape ({SEQUENCE_LENGTH}, 225), got {keypoints_array.shape}")
                continue
                
            # Apply the same standardization as training, if available
            if scaler is not None:
                keypoints_array = scaler.transform(keypoints_array)

            prediction = model.predict(keypoints_array[np.newaxis, :, :], verbose=0)
            
            # Get prediction details
            max_prob = np.amax(prediction)
            predicted_class = np.argmax(prediction)
            action = actions[predicted_class]
            
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
                else:
                    print(f"Prediction not consistent: {action} appears {recent_actions.count(action)}/3 times")
            else:
                print(f"Low confidence: {max_prob:.3f} < {CONFIDENCE_THRESHOLD}")

            if len(sentence) > 5:
                sentence = sentence[-5:]

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
            print(f"📝 Word buffer: {word_buffer}")
        elif sentence:
            display_text = sentence[-1] if sentence else "No sentence yet"
            print(f"📝 Current sentence: {display_text}")
        else:
            display_text = "Press SPACE to start collecting words"
            print("Waiting for SPACE to start...")
        text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x_pos = (image.shape[1] - text_size[0]) // 2 if text_size[0] < image.shape[1] else 0
        y_pos = image.shape[0] - 40

        image = draw_armenian_text(image, display_text, (x_pos, y_pos), font_size=36, color=(255, 255, 0))

        cv2.imshow('Real-Time Sign Language Recognition', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()