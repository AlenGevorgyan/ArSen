import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import joblib
from tqdm import tqdm

# ====================== CONFIG ======================
PATH = os.path.join("dataset2")
frames = 30  # Updated sequence length
# ====================================================

# Load class labels (actions)
actions = np.array(sorted([d for d in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, d))]))
label_map = {label: num for num, label in enumerate(actions)}

landmarks, labels = [], []

# ====================== LOAD DATA ======================
for action in tqdm(actions, desc="Loading Actions"):
    action_path = os.path.join(PATH, action)
    sequence_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]

    for sequence_folder in sequence_folders:
        temp_frames = []
        sequence_path = os.path.join(action_path, sequence_folder)

        for frame_num in range(frames):
            frame_file = f"frame_{frame_num:05d}.npy"
            npy_path = os.path.join(sequence_path, frame_file)

            if os.path.exists(npy_path):
                npy = np.load(npy_path)
                if npy.shape[0] == 225:
                    temp_frames.append(npy)
                else:
                    print(f"⚠️ Skipping {npy_path}: Expected 225 features, got {npy.shape[0]}")
                    temp_frames = []
                    break
            else:
                print(f"⚠️ Missing {npy_path}. Skipping sequence.")
                temp_frames = []
                break

        if len(temp_frames) == frames:
            landmarks.append(temp_frames)
            labels.append(label_map[action])
# ======================================================

# Convert to numpy arrays
X = np.array(landmarks)
Y = to_categorical(labels, num_classes=len(actions)).astype(int)

if X.shape[0] == 0:
    raise ValueError("❌ No data loaded. Check dataset structure.")

print(f"\n✅ Loaded {X.shape[0]} sequences.")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# ====================== SPLIT DATA ======================
y_indices = np.argmax(Y, axis=1)
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.20, random_state=34, stratify=y_indices, shuffle=True
)
y_temp_indices = np.argmax(Y_temp, axis=1)
X_test, X_val, Y_test, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.30, random_state=34, stratify=y_temp_indices, shuffle=True
)
# ========================================================

# ====================== SCALING ======================
scaler = StandardScaler()
n_train, t_train, f_train = X_train.shape
scaler.fit(X_train.reshape(n_train * t_train, f_train))

def scale_split(arr):
    n, t, f = arr.shape
    return scaler.transform(arr.reshape(n * t, f)).reshape(n, t, f)

X_train = scale_split(X_train)
X_val = scale_split(X_val)
X_test = scale_split(X_test)

joblib.dump(scaler, 'feature_scaler.pkl')
np.save('actions.npy', actions)
# =====================================================

# ====================== MODEL ======================
input_feature_count = X.shape[2]
print(f"Input feature count: {input_feature_count}")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(frames, input_feature_count), dropout=0.2, recurrent_dropout=0.2),
    BatchNormalization(),

    LSTM(192, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
    BatchNormalization(),

    LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
    BatchNormalization(),

    Dense(128, activation='relu'),
    Dropout(0.35),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(actions.shape[0], activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.summary()
# ===================================================

# ====================== TRAIN ======================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max')
]

y_train_labels = np.argmax(Y_train, axis=1)
class_weights_array = class_weight.compute_class_weight(class_weight='balanced',
                                                        classes=np.arange(actions.shape[0]),
                                                        y=y_train_labels)
class_weights = {i: w for i, w in enumerate(class_weights_array)}

print(f"\n🚀 Starting training with {len(X_train)} samples...")
history = model.fit(
    X_train,
    Y_train,
    epochs=250,
    validation_data=(X_val, Y_val),
    callbacks=callbacks,
    batch_size=64,
    class_weight=class_weights,
    verbose=1
)
# ===================================================

# ====================== EVALUATE ======================
model.save('my_model.h5')
predictions = np.argmax(model.predict(X_test), axis=1)
test_labels = np.argmax(Y_test, axis=1)

accuracy = metrics.accuracy_score(test_labels, predictions)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(metrics.classification_report(test_labels, predictions, target_names=actions))

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    cm = metrics.confusion_matrix(test_labels, predictions)
    fig, ax = plt.subplots(figsize=(max(6, len(actions) * 0.4), max(4, len(actions) * 0.4)))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=actions, yticklabels=actions, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close(fig)
    print("✅ Saved confusion_matrix.png")
except Exception as e:
    print(f"⚠️ Skipping confusion matrix plot: {e}")
