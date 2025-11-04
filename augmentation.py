import cv2 as cv
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# === CONFIG ===
VIDEO_ROOT = r"videos"
SAVE_ROOT = r"dataset"
SEQUENCE_LENGTH = 20

# --- Augmentation Config ---
# 1. Scaling (your original augmentation)
#    This makes the fingers longer/shorter relative to their base knuckle
SCALE_VALUES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

# 2. Jitter (new augmentation)
#    This adds small random noise to hand keypoints to simulate tiny variations
JITTER_STRENGTHS = [0.002, 0.005]  # Small std dev for random noise

# === INIT ===
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# === KEYPOINT EXTRACTION (Simplified) ===

def extract_keypoints(results):
    """Extract keypoints and return concatenated 225 features (pose + left hand + right hand)"""
    # 99 features for pose
    pose = np.array([[lm.x, lm.y, lm.z] for lm in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)

    # 63 features for left hand
    lh = np.array([[lm.x, lm.y, lm.z] for lm in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # 63 features for right hand
    rh = np.array([[lm.x, lm.y, lm.z] for lm in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)

    # Total = 99 + 63 + 63 = 225 features
    return np.concatenate([pose, lh, rh])


# === AUGMENTATION FUNCTIONS ===

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


def apply_scaling(sequence, scale_factor):
    """
    Applies the finger-scaling augmentation to an entire sequence.
    'sequence' must be a numpy array of shape (SEQUENCE_LENGTH, 225)
    """
    if scale_factor == 1.0:
        return sequence

    augmented_sequence = []
    finger_groups = [
        range(1, 5), range(5, 9), range(9, 13), range(13, 17), range(17, 21),
    ]

    for frame_keypoints in sequence:
        pose = frame_keypoints[:99]
        lh_flat = frame_keypoints[99:162]
        rh_flat = frame_keypoints[162:225]

        # Only augment if hands are present
        if np.all(lh_flat == 0) and np.all(rh_flat == 0):
            augmented_sequence.append(frame_keypoints)
            continue

        lh = reshape_hand(lh_flat)
        rh = reshape_hand(rh_flat)

        for group in finger_groups:
            if not np.all(lh == 0):
                lh[list(group)] = scale_finger(lh[list(group)], scale_factor)
            if not np.all(rh == 0):
                rh[list(group)] = scale_finger(rh[list(group)], scale_factor)

        lh = clip_coords(lh)
        rh = clip_coords(rh)

        new_keypoints = np.concatenate([pose, flatten_hand(lh), flatten_hand(rh)])
        augmented_sequence.append(new_keypoints)

    return np.array(augmented_sequence)


def apply_jitter(sequence, strength):
    """
    Applies jitter (random noise) to the hand keypoints in a sequence.
    'sequence' must be a numpy array of shape (SEQUENCE_LENGTH, 225)
    """
    if strength == 0.0:
        return sequence

    # Create noise with the same shape as the sequence
    noise = np.random.normal(0.0, strength, sequence.shape)

    # Zero out the noise for the pose keypoints (indices 0-98)
    noise[:, :99] = 0.0

    return sequence + noise


# === MAIN PROCESSING LOOP (Refactored for Efficiency) ===
def main():
    """
    Processes each video ONCE, then applies all augmentations in memory
    before saving each augmented sequence as a SINGLE .npy file.
    """
    video_paths_to_process = []
    video_root_path = Path(VIDEO_ROOT)
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_paths_to_process.extend(video_root_path.rglob(ext))

    print(f"Found {len(video_paths_to_process)} videos to process.")

    for video_path in video_paths_to_process:
        class_name = video_path.parent.name
        video_name = video_path.stem
        print(f"\n🎥 Processing video: {video_path}")

        # --- 1. Extract Keypoints (Read video ONLY ONCE) ---
        cap = cv.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"    ⚠️ Error: Could not open video. Skipping.")
            continue

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        original_keypoints_sequence = []

        if total_frames == 0:
            print(f"    ⚠️ Error: Video has 0 frames. Skipping.")
            cap.release()
            continue

        # Get frame indices to sample
        indices_to_process = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH,
                                         dtype=int) if total_frames > SEQUENCE_LENGTH else np.arange(total_frames)

        for frame_idx in indices_to_process:
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                original_keypoints_sequence.append(np.zeros(225))
                continue

            try:
                image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic.process(image_rgb)
                keypoints = extract_keypoints(results)
                original_keypoints_sequence.append(keypoints)
            except Exception as e:
                print(f"    Error processing frame {frame_idx}: {e}")
                original_keypoints_sequence.append(np.zeros(225))

        cap.release()

        # --- 2. Pad Sequence (if video was shorter than SEQUENCE_LENGTH) ---
        if not original_keypoints_sequence:
            print(f"    ⚠️ Error: No keypoints extracted. Skipping.")
            continue

        while len(original_keypoints_sequence) < SEQUENCE_LENGTH:
            original_keypoints_sequence.append(original_keypoints_sequence[-1])  # Pad with last frame

        # Convert to a single numpy array
        try:
            base_sequence = np.array(original_keypoints_sequence)
        except ValueError as e:
            print(f"    ⚠️ Error: Mismatch in keypoint array shapes. Skipping. Details: {e}")
            continue

        # --- 3. Apply Augmentations & Save (Smarter Storage) ---
        save_dir = Path(SAVE_ROOT) / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # A. Apply SCALING augmentations
        for scale_factor in SCALE_VALUES:
            # Generate the new sequence
            augmented_sequence = apply_scaling(base_sequence, scale_factor)

            # Define the single .npy file path
            sequence_name = f"{video_name}_scale_{scale_factor:.1f}.npy"
            save_path = save_dir / sequence_name

            # Save the ENTIRE sequence (20, 225) as one file
            np.save(save_path, augmented_sequence)

        # B. Apply JITTER augmentations (based on the original 1.0 scale)
        for strength in JITTER_STRENGTHS:
            # We jitter the *original* (scale=1.0) sequence
            augmented_sequence = apply_jitter(base_sequence, strength)

            sequence_name = f"{video_name}_jitter_{strength:.3f}.npy"
            save_path = save_dir / sequence_name
            np.save(save_path, augmented_sequence)

        print(f"    ✅ Saved {len(SCALE_VALUES) + len(JITTER_STRENGTHS)} augmentations for {video_name}")

    print("\nAll videos have been processed and augmented!")


if __name__ == "__main__":
    main()