"""
Visualization tool for Sign Language Recognition System
Generates training curves, dataset statistics, and model performance visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from train_model import SignLanguageGRU, get_class_mapping
from sklearn.metrics import confusion_matrix, classification_report
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_training_history(history_file='training_history.json'):
    """Load training history from JSON file."""
    if not os.path.exists(history_file):
        print(f"⚠️  Training history file '{history_file}' not found.")
        print("   Run train_model.py first to generate training metrics.")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    return history


def plot_training_curves(history):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss difference (overfitting indicator)
    loss_diff = [val - train for train, val in zip(history['train_loss'], history['val_loss'])]
    axes[1, 1].plot(epochs, loss_diff, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Overfitting Indicator (Val Loss - Train Loss)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("✅ Saved training_curves.png")
    plt.close()


def plot_dataset_statistics(data_dir='dataset'):
    """Plot dataset statistics including class distribution."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"⚠️  Dataset directory '{data_dir}' not found.")
        return
    
    class_to_idx = get_class_mapping(data_dir)
    class_names = list(class_to_idx.keys())
    class_counts = []
    
    for class_name in class_names:
        class_dir = data_path / class_name
        count = len(list(class_dir.glob("*.npy")))
        class_counts.append(count)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    axes[0].barh(class_names, class_counts, color='steelblue')
    axes[0].set_title('Number of Samples per Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Samples')
    axes[0].set_ylabel('Class')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Pie chart (top 10 classes)
    sorted_indices = np.argsort(class_counts)[-10:]
    top_classes = [class_names[i] for i in sorted_indices]
    top_counts = [class_counts[i] for i in sorted_indices]
    
    axes[1].pie(top_counts, labels=top_classes, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Top 10 Classes by Sample Count', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=300, bbox_inches='tight')
    print("✅ Saved dataset_statistics.png")
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total classes: {len(class_names)}")
    print(f"   Total samples: {sum(class_counts)}")
    print(f"   Average samples per class: {np.mean(class_counts):.1f}")
    print(f"   Min samples: {min(class_counts)} ({class_names[np.argmin(class_counts)]})")
    print(f"   Max samples: {max(class_counts)} ({class_names[np.argmax(class_counts)]})")
    
    plt.close()


def plot_confusion_matrix(model_path='best_model.pth', data_dir='dataset', device='cpu'):
    """Generate confusion matrix from model predictions."""
    if not os.path.exists(model_path):
        print(f"⚠️  Model file '{model_path}' not found.")
        print("   Train the model first using train_model.py")
        return
    
    print("\n🔄 Generating confusion matrix...")
    
    # Load model
    class_to_idx = get_class_mapping(data_dir)
    num_classes = len(class_to_idx)
    class_names = list(class_to_idx.keys())
    
    model = SignLanguageGRU(input_size=450, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and predict on validation set
    from train_model import SignLanguageDataset, preprocess_keypoints
    from torch.utils.data import DataLoader
    
    dataset = SignLanguageDataset(data_dir, class_to_idx, preprocess_keypoints, seq_len=20)
    
    # Use 20% of data for evaluation
    from torch.utils.data import random_split
    _, val_dataset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(max(12, num_classes * 0.5), max(10, num_classes * 0.5)))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved confusion_matrix.png")
    
    # Print classification report
    print("\n📈 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    plt.close()


def plot_model_architecture():
    """Visualize model architecture."""
    print("\n📐 Model Architecture:")
    print("=" * 60)
    
    # Create a simple text-based visualization
    architecture_text = """
    Input: (batch_size, 20, 450)
         ↓
    Bidirectional GRU (2 layers, 128 hidden units)
         ↓
    Attention Mechanism (learns important frames)
         ↓
    BatchNorm1d
         ↓
    FC Layer (128 → 64) + ReLU + Dropout(0.4)
         ↓
    FC Layer (64 → num_classes)
         ↓
    Output: (batch_size, num_classes)
    """
    
    print(architecture_text)
    
    # Calculate approximate parameters
    input_size = 450
    hidden_size = 128
    num_layers = 2
    num_classes = 26  # Example
    
    # GRU parameters (bidirectional)
    gru_params = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size) * num_layers * 2
    attention_params = hidden_size * 2  # weights
    fc1_params = (hidden_size * 2) * 64 + 64
    fc2_params = 64 * num_classes + num_classes
    
    total_params = gru_params + attention_params + fc1_params + fc2_params
    
    print(f"📊 Approximate Model Parameters: {total_params:,}")
    print(f"   GRU: {gru_params:,}")
    print(f"   Attention: {attention_params:,}")
    print(f"   FC1: {fc1_params:,}")
    print(f"   FC2: {fc2_params:,}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics and dataset statistics")
    parser.add_argument("--history", type=str, default="training_history.json", 
                       help="Path to training history JSON file")
    parser.add_argument("--data_dir", type=str, default="dataset",
                       help="Path to dataset directory")
    parser.add_argument("--model", type=str, default="best_model.pth",
                       help="Path to trained model file")
    parser.add_argument("--all", action="store_true",
                       help="Generate all visualizations")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only generate dataset statistics (no training history needed)")
    
    args = parser.parse_args()
    
    print("🎨 Sign Language Recognition - Visualization Tool")
    print("=" * 60)
    
    generated_files = []
    
    # Always try to generate dataset statistics (doesn't require training)
    print("\n📊 Analyzing dataset statistics...")
    try:
        plot_dataset_statistics(args.data_dir)
        if os.path.exists('dataset_statistics.png'):
            generated_files.append('dataset_statistics.png')
    except Exception as e:
        print(f"⚠️  Error generating dataset statistics: {e}")
    
    # Always show model architecture
    print("\n📐 Displaying model architecture...")
    plot_model_architecture()
    
    # Load training history (optional)
    history = load_training_history(args.history)
    
    # Generate training curves if history exists
    if history:
        print("\n📈 Generating training curves...")
        try:
            plot_training_curves(history)
            if os.path.exists('training_curves.png'):
                generated_files.append('training_curves.png')
        except Exception as e:
            print(f"⚠️  Error generating training curves: {e}")
    else:
        print("\n⚠️  Skipping training curves (no history file)")
        print("   Run 'python train_model.py' first to generate training metrics")
    
    # Generate confusion matrix if model exists
    if args.all or (not args.stats_only and os.path.exists(args.model)):
        print("\n🎯 Generating confusion matrix...")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            plot_confusion_matrix(args.model, args.data_dir, device)
            if os.path.exists('confusion_matrix.png'):
                generated_files.append('confusion_matrix.png')
        except Exception as e:
            print(f"⚠️  Error generating confusion matrix: {e}")
            print(f"   Make sure the model file '{args.model}' exists and is trained")
    
    print("\n✅ Visualization complete!")
    
    if generated_files:
        print("\n📁 Generated files:")
        for file in generated_files:
            if os.path.exists(file):
                print(f"  ✅ {file}")
    else:
        print("\n⚠️  No visualization files were generated.")
        print("   This is normal if:")
        print("   - Training hasn't been run yet (no training_history.json)")
        print("   - Model file doesn't exist (no confusion matrix)")
        print("   - Dataset directory is missing or empty")
        print("\n   To generate visualizations:")
        print("   1. Run 'python train_model.py' to train the model")
        print("   2. Then run 'python visualize.py --all' for all visualizations")


if __name__ == "__main__":
    main()

