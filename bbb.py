import numpy as np
import cv2
import mediapipe as mp
import os

# === CONFIG ===
base_dir = r"videos"  # Use your actual dataset directory
fps = 5  # кадров в секунду (чем меньше — тем медленнее)

# === INIT ===
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# === UTILS ===
def reshape_hand(flat):
    """(63,) -> (21,3)"""
    return flat.reshape((21, 3))

def reshape_pose(flat):
    """(99,) -> (33,3)"""
    return flat.reshape((33, 3))

def draw_skeleton(image, lh, rh, pose, color_offset=(0, 0, 0)):
    """Рисует руки и позу на изображении."""
    h, w, _ = image.shape
    def to_pixel(p): return int(p[0]*w), int(p[1]*h)

    # тело
    for start, end in POSE_CONNECTIONS:
        p1, p2 = pose[start], pose[end]
        cv2.line(image, to_pixel(p1), to_pixel(p2), (0, 255, 0), 2)
    for p in pose:
        cv2.circle(image, to_pixel(p), 3, (0, 200, 255), -1)

    # руки
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, to_pixel(lh[start]), to_pixel(lh[end]), (255, 0, 0), 2)
        cv2.line(image, to_pixel(rh[start]), to_pixel(rh[end]), (0, 0, 255), 2)
    for p in lh:
        cv2.circle(image, to_pixel(p), 3, (255, 100, 0), -1)
    for p in rh:
        cv2.circle(image, to_pixel(p), 3, (0, 100, 255), -1)


# === MAIN ===
# Check if base directory exists
if not os.path.exists(base_dir):
    print(f"❌ Directory '{base_dir}' not found")
    print("Available directories:")
    for item in os.listdir("."):
        if os.path.isdir(item):
            print(f"  - {item}")
    exit()

# Automatically detect available actions
available_actions = []
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path):
        # Check if it's a dataset directory (has .npy files)
        has_npy_files = any(f.endswith('.npy') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f)))
        if has_npy_files:
            available_actions.append(item)
        else:
            # Check if it has subdirectories with .npy files
            subdirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
            for subdir in subdirs:
                subdir_path = os.path.join(item_path, subdir)
                if any(f.endswith('.npy') for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))):
                    available_actions.append(item)
                    break

if not available_actions:
    print(f"❌ No action directories with .npy files found in '{base_dir}'")
    print("Looking for directories with .npy files...")
    exit()

print(f"🔹 Available actions: {available_actions}")

# Use first 4 actions for comparison (or all if less than 4)
actions_to_compare = available_actions[:4]
print(f"🔹 Comparing actions: {actions_to_compare}")

# Find the first available sequence with frames
files = []
first_sequence_path = None

for action in actions_to_compare:
    action_path = os.path.join(base_dir, action)
    
    # Look for .npy files directly in action directory
    direct_files = [f for f in os.listdir(action_path) if f.endswith('.npy') and os.path.isfile(os.path.join(action_path, f))]
    if direct_files:
        first_sequence_path = action_path
        files = sorted(direct_files)
        break
    
    # Look for .npy files in subdirectories
    sequence_folders = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
    for sequence_folder in sequence_folders:
        sequence_path = os.path.join(action_path, sequence_folder)
        sequence_files = [f for f in os.listdir(sequence_path) if f.endswith('.npy') and os.path.isfile(os.path.join(sequence_path, f))]
        if sequence_files:
            first_sequence_path = sequence_path
            files = sorted(sequence_files)
            break
    
    if files:
        break

if not files:
    print(f"❌ No .npy frame files found in any action directory")
    exit()

print(f"🔹 Found {len(files)} frames in '{os.path.basename(first_sequence_path)}'")

for filename in files:
    frames = []
    for action in actions_to_compare:
        action_path = os.path.join(base_dir, action)
        npy_path = None
        
        # Try to find the frame file in this action
        # First check if files are directly in action directory
        direct_path = os.path.join(action_path, filename)
        if os.path.exists(direct_path):
            npy_path = direct_path
        else:
            # Look in subdirectories
            sequence_folders = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
            for sequence_folder in sequence_folders:
                sequence_path = os.path.join(action_path, sequence_folder)
                potential_path = os.path.join(sequence_path, filename)
                if os.path.exists(potential_path):
                    npy_path = potential_path
                    break
        
        if not npy_path or not os.path.exists(npy_path):
            print(f"⚠️ Frame {filename} not found for action '{action}'")
            continue

        try:
            data = np.load(npy_path)
            if data.ndim == 1:
                sample = data
            else:
                sample = data[0]

            # Check if we have the right number of features (225)
            if len(sample) != 225:
                print(f"⚠️ Wrong feature count in {filename}: {len(sample)} (expected 225)")
                continue

            lh = reshape_hand(sample[:63])
            rh = reshape_hand(sample[63:126])
            pose = reshape_pose(sample[126:225])

            frame = np.zeros((720, 480, 3), dtype=np.uint8)
            draw_skeleton(frame, lh, rh, pose)
            
            # Create a cleaner label
            sequence_name = os.path.basename(os.path.dirname(npy_path)) if os.path.dirname(npy_path) != action_path else "direct"
            label = f"{action}"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            frames.append(frame)
            
        except Exception as e:
            print(f"⚠️ Error processing {npy_path}: {e}")
            continue

    if not frames:
        continue

    # соединяем все картинки горизонтально
    combined = cv2.hconcat(frames)
    cv2.imshow("Comparison of Scales", combined)

    key = cv2.waitKey(int(1000/fps))
    if key == 27:
        break

cv2.destroyAllWindows()
print("✅ Визуализация завершена.")
