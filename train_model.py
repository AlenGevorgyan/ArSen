import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import random


# === 1. REPRODUCIBILITY ===
def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Makes CUDA operations deterministic, but can be a bit slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# === 2. THE MODEL (SignLanguageGRU) ===
# (This is the improved model we discussed)

class Attention(nn.Module):
    """Simple dot-product attention."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weights = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        scores = torch.matmul(x, self.weights)  # (batch_size, seq_len)
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
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

        gru_output_size = hidden_size * 2  # Bidirectional
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


# === 3. PREPROCESSING FUNCTION (CRITICAL!) ===

# --- MODIFIED: Added seq_len as an argument ---
def preprocess_keypoints(sequence, seq_len):
    """
    Applies normalization and motion deltas to the raw keypoint sequence.
    Input shape: (20, 225)
    Output shape: (20, 450)
    """
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(sequence)

    # --- A. Normalization (Translation Invariance) ---
    # Reshape to (seq_len, num_groups, num_keypoints, num_coords)
    # --- MODIFIED: Use seq_len argument instead of global variable ---
    pose = sequence[:, :99].reshape(seq_len, 33, 3)
    lh = sequence[:, 99:162].reshape(seq_len, 21, 3)
    rh = sequence[:, 162:225].reshape(seq_len, 21, 3)

    # Get reference points (wrist for hands, nose for pose)
    # Use [:, 0:1, :] to keep the dimension for broadcasting
    nose = pose[:, 0:1, :]
    lwrist = lh[:, 0:1, :]
    rwrist = rh[:, 0:1, :]

    # Subtract the reference point. Zeros (missing) will just subtract zero.
    pose_norm = pose - nose
    lh_norm = lh - lwrist
    rh_norm = rh - rwrist

    # Flatten back to (20, 225)
    # --- MODIFIED: Use seq_len argument instead of global variable ---
    normalized_sequence = np.concatenate([
        pose_norm.reshape(seq_len, 99),
        lh_norm.reshape(seq_len, 63),
        rh_norm.reshape(seq_len, 63)
    ], axis=1)

    # --- B. Motion Deltas (Velocity) ---
    # Calculate frame-to-frame differences
    deltas = np.diff(normalized_sequence, axis=0)

    # Pad the first frame with zeros
    deltas = np.concatenate([np.zeros((1, 225)), deltas], axis=0)

    # --- C. Concatenate Features ---
    # Final shape: (20, 450)
    final_features = np.concatenate([normalized_sequence, deltas], axis=1)

    return final_features


# === 4. DATA LOADER ===

def get_class_mapping(data_root):
    """Finds all class folders and creates a name-to-index mapping."""
    data_path = Path(data_root)
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return class_to_idx


class SignLanguageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading single .npy sequence files.
    """

    # --- MODIFIED: Added seq_len to __init__ ---
    def __init__(self, data_root, class_to_idx, preprocess_fn, seq_len):
        self.data_root = Path(data_root)
        self.class_to_idx = class_to_idx
        self.preprocess_fn = preprocess_fn
        self.seq_len = seq_len  # --- Store seq_len ---

        self.filepaths = []
        self.labels = []

        # Load all file paths and labels
        for class_name, label_idx in self.class_to_idx.items():
            class_dir = self.data_root / class_name
            for seq_file in class_dir.glob("*.npy"):
                self.filepaths.append(seq_file)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]

        # Load the (20, 225) sequence
        raw_sequence = np.load(path)

        # Apply preprocessing (Normalization + Deltas)
        # --- MODIFIED: Pass self.seq_len to the preprocess function ---
        processed_sequence = self.preprocess_fn(raw_sequence, self.seq_len)

        return torch.tensor(processed_sequence, dtype=torch.float32), \
            torch.tensor(label, dtype=torch.long)


# === 5. TRAIN & VALIDATE FUNCTIONS ===

def train_fn(loader, model, optimizer, criterion, device):
    """One epoch of training."""
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0

    loop = tqdm(loader, desc="Training")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels)
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * corrects / total
    return epoch_loss, epoch_acc


def val_fn(loader, model, criterion, device):
    """One epoch of validation."""
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0

    loop = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels)
            total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * corrects / total
    return epoch_loss, epoch_acc


# === 6. MAIN ORCHESTRATION ===

def main(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data Setup ---
    print("Loading data...")
    class_to_idx = get_class_mapping(args.data_dir)
    num_classes = len(class_to_idx)

    # --- MODIFIED: Pass args.seq_len to the Dataset constructor ---
    dataset = SignLanguageDataset(
        args.data_dir,
        class_to_idx,
        preprocess_keypoints,
        args.seq_len
    )

    # --- Train/Val Split ---
    val_percent = 0.2
    val_size = int(len(dataset) * val_percent)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- Model, Loss, Optimizer ---
    # The input size is 450 (225 normalized + 225 deltas)
    # --- MODIFIED: Removed global SEQUENCE_LENGTH lines ---

    model = SignLanguageGRU(
        input_size=450,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        num_gru_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Scheduler: Reduce LR if val_loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # --- Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        train_loss, train_acc = train_fn(train_loader, model, optimizer, criterion, device)
        val_loss, val_acc = val_fn(val_loader, model, criterion, device)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Step the scheduler
        scheduler.step(val_loss)

        # --- Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = "laptop.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Recognition Training Pipeline")

    # --- Paths and Data ---
    parser.add_argument("--data_dir", type=str, default="dataset", help="Root directory of the processed dataset")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length (number of frames)")

    # --- Training Params ---
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # --- Model Params ---
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the GRU")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability")

    args = parser.parse_args()

    main(args)
