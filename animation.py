import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import mediapipe as mp

# === CONFIG ===
data_folder = r"D:\Arsen_AI\dataset\բոլորը\WIN_20251019_16_01_41_Pro_scale_1.0"
frame_files = sorted(glob.glob(os.path.join(data_folder, "frame_*.npy")))

mp_pose = mp.solutions.pose
pose_connections = mp_pose.POSE_CONNECTIONS

# === FIGURE SETUP ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# invert Y axis to match MediaPipe coordinate system
ax.invert_yaxis()

# === ANIMATION FUNCTION ===
def update_frame(frame_idx):
    ax.cla()  # clear previous
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_yaxis()

    # load the first augmented sample from this frame
    data = np.load(frame_files[frame_idx])[0]  # shape=(63+63+99,)

    # split hands and pose
    lh = data[:63].reshape((21,3))
    rh = data[63:126].reshape((21,3))
    pose = data[126:].reshape((33,3))

    # draw pose skeleton
    for connection in pose_connections:
        start, end = connection
        x = [pose[start,0], pose[end,0]]
        y = [pose[start,1], pose[end,1]]
        z = [pose[start,2], pose[end,2]]
        ax.plot(x, y, z, c='blue', linewidth=1)

    # draw hand landmarks as scatter
    ax.scatter(lh[:,0], lh[:,1], lh[:,2], c='green', s=20, label='Left Hand')
    ax.scatter(rh[:,0], rh[:,1], rh[:,2], c='orange', s=20, label='Right Hand')
    ax.scatter(pose[:,0], pose[:,1], pose[:,2], c='red', s=10, label='Pose')

    ax.set_title(f"Frame {frame_idx+1}/{len(frame_files)}")

# === CREATE ANIMATION ===
from matplotlib.animation import FuncAnimation

ani = FuncAnimation(fig, update_frame, frames=len(frame_files), interval=50)
plt.show()
