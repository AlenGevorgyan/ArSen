import cv2 as cv
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# === CONFIG ===
VIDEO_ROOT = r"videos"

SAVE_ROOT = r"dataset2"

SEQUENCE_LENGTH = 30

SCALE_VALUES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

# === INIT ===
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# === UTILITIES (Unchanged) ===

def extract_keypoints(results):
    """Extract keypoints and return concatenated 225 features (pose + left hand + right hand)"""
    # Extract pose landmarks (33 points * 3 coordinates = 99 features)
    pose = np.array([[lm.x, lm.y, lm.z] for lm in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)

    # Extract left hand landmarks (21 points * 3 coordinates = 63 features)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # Extract right hand landmarks (21 points * 3 coordinates = 63 features)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    # CONCATENATE THE ARRAYS INTO ONE 225-FEATURE ARRAY
    keypoints = np.concatenate([pose, lh, rh])
    
    # Ensure we have exactly 225 features
    if len(keypoints) != 225:
        print(f"Warning: Expected 225 features, got {len(keypoints)}")
        # Pad with zeros or truncate as needed
        if len(keypoints) < 225:
            keypoints = np.pad(keypoints, (0, 225 - len(keypoints)), 'constant')
        else:
            keypoints = keypoints[:225]
    
    return keypoints


def reshape_hand(flat):
    return flat.reshape((21, 3))


def flatten_hand(mat):
    return mat.flatten()


def scale_finger(finger, scale):
    base = finger[0]
    return base + (finger - base) * scale


def clip_coords(hand3d):
    hand3d[:, 0] = np.clip(hand3d[:, 0], 0, 1)
    hand3d[:, 1] = np.clip(hand3d[:, 1], 0, 1)
    hand3d[:, 2] = np.clip(hand3d[:, 2], -1, 1)
    return hand3d


def scale_keypoints(keypoints, scale_factor):
    """Scale hand keypoints while keeping pose unchanged"""
    if scale_factor == 1.0:
        return keypoints

    pose = keypoints[:99]
    lh_flat = keypoints[99:162]
    rh_flat = keypoints[162:225]

    lh = reshape_hand(lh_flat)
    rh = reshape_hand(rh_flat)

    if np.all(lh == 0) and np.all(rh == 0):
        return keypoints

    finger_groups = [
        range(1, 5), range(5, 9), range(9, 13), range(13, 17), range(17, 21),
    ]

    for group in finger_groups:
        if not np.all(lh == 0):
            lh[list(group)] = scale_finger(lh[list(group)], scale_factor)
        if not np.all(rh == 0):
            rh[list(group)] = scale_finger(rh[list(group)], scale_factor)

    lh = clip_coords(lh)
    rh = clip_coords(rh)

    return np.concatenate([pose, flatten_hand(lh), flatten_hand(rh)])


# === MAIN PROCESSING LOOP ===
def main():
    """
    Processes each video multiple times with different scale factors to augment the dataset.
    """
    # Find all video files first
    video_paths_to_process = []
    for dirpath, _, filenames in os.walk(VIDEO_ROOT):
        for filename in filenames:
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths_to_process.append(os.path.join(dirpath, filename))

    print(f"Found {len(video_paths_to_process)} videos to process.")

    # Process each video found
    for video_path in video_paths_to_process:
        class_name = os.path.basename(os.path.dirname(video_path))
        video_name = Path(video_path).stem

        print(f"\n🎥 Augmenting video: {video_path}")

        # Loop through each scale factor to create a new variant
        for scale_factor in SCALE_VALUES:

            # --- 1. Define Paths for this variant ---
            sequence_name = f"{video_name}_scale_{scale_factor:.1f}"
            save_dir = os.path.join(SAVE_ROOT, class_name, sequence_name)
            os.makedirs(save_dir, exist_ok=True)

            print(f"  → Generating variant with scale={scale_factor:.1f}")

            # --- 2. Extract Frames ---
            cap = cv.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"    ⚠️ Error: Could not open video file. Skipping this variant.")
                continue

            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            sequence_keypoints = []

            indices_to_process = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH,
                                             dtype=int) if total_frames > SEQUENCE_LENGTH else range(total_frames)

            for frame_idx in indices_to_process:
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"    ⚠️ Warning: Could not read frame {frame_idx}")
                    continue

                try:
                    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = holistic.process(image_rgb)

                    # Check if MediaPipe detected anything
                    if not results.pose_landmarks and not results.left_hand_landmarks and not results.right_hand_landmarks:
                        print(f"    ⚠️ Warning: No landmarks detected in frame {frame_idx}")
                        # Use zeros as fallback
                        keypoints = np.zeros(225)
                    else:
                        keypoints = extract_keypoints(results)

                    # Validate keypoints
                    if len(keypoints) != 225:
                        print(f"    ⚠️ Warning: Invalid keypoints length {len(keypoints)}, expected 225")
                        keypoints = np.zeros(225)  # Use zeros as fallback

                    # Scale the keypoints
                    scaled_keypoints = scale_keypoints(keypoints, scale_factor)
                    sequence_keypoints.append(scaled_keypoints)

                except Exception as e:
                    print(f"    ⚠️ Error processing frame {frame_idx}: {e}")
                    # Use zeros as fallback
                    sequence_keypoints.append(np.zeros(225))

            cap.release()

            # --- 3. Pad & Save ---
            if len(sequence_keypoints) > 0:
                while len(sequence_keypoints) < SEQUENCE_LENGTH:
                    sequence_keypoints.append(sequence_keypoints[-1])

                for i, keypoints in enumerate(sequence_keypoints):
                    filename = f"frame_{i:05d}.npy"
                    path = os.path.join(save_dir, filename)
                    np.save(path, keypoints)
                print(f"    ✅ Saved {len(sequence_keypoints)} frames to {save_dir}")
            else:
                print(f"    ⚠️ Warning: No keypoints extracted for this variant. Skipping.")

    print("\nAll videos have been augmented!")


if __name__ == "__main__":
    main()