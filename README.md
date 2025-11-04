# Sign Language Translator Neural Network

This project implements a real-time sign language recognition system using MediaPipe for pose and hand tracking, and TensorFlow/Keras for the neural network model.

## Project Structure

- `main.py` - Real-time inference application
- `train_model.py` - Model training script
- `preproccess.py` - Data collection script for creating training data
- `my_functions.py` - Utility functions for keypoint extraction and image processing
- `aaa.py` - Video augmentation script for creating scaled datasets
- `bbb.py` - Visualization script for comparing different scales
- `dataset/` - Training data directory with organized sequences

## Fixed Issues

### 1. Keypoint Extraction
- **Problem**: The original keypoint extraction only included hand landmarks (126 features)
- **Solution**: Updated to include pose + left hand + right hand landmarks (225 features total)
- **Files affected**: `my_functions.py`, `main.py`

### 2. Model Architecture
- **Problem**: Model was hardcoded to expect 225 features but data loading wasn't consistent
- **Solution**: Dynamic input shape based on actual data, improved LSTM architecture
- **Files affected**: `train_model.py`

### 3. Data Loading
- **Problem**: Inconsistent data format and missing validation
- **Solution**: Added proper validation for 225 features per frame, 20 frames per sequence
- **Files affected**: `train_model.py`

### 4. Error Handling
- **Problem**: Limited error handling and debugging information
- **Solution**: Added comprehensive validation and error messages
- **Files affected**: All main files

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Collection
To collect new training data:
```bash
python preproccess.py
```
- Press spacebar to start recording each sequence
- The script will record 20 frames for each sign
- Data is saved in the `dataset/` directory

### 2. Model Training
To train the neural network:
```bash
python train_model.py
```
- Loads data from the `dataset/` directory
- Trains an LSTM model with early stopping
- Saves the best model as `my_model.h5`

### 3. Real-time Inference
To run the real-time sign language recognition:
```bash
python main.py
```
- Uses your webcam for real-time detection
- Press spacebar to clear the sentence
- Press 'q' to quit

## Data Format

Each training sequence consists of:
- 20 frames per sequence
- 225 features per frame:
  - 99 features from pose landmarks (33 points × 3 coordinates)
  - 63 features from left hand landmarks (21 points × 3 coordinates)
  - 63 features from right hand landmarks (21 points × 3 coordinates)

## Model Architecture

- Input: (20, 225) - 20 frames with 225 features each
- LSTM layers: 64 → 128 → 64 units
- Dense layers: 128 → 64 → num_classes
- Output: Softmax probabilities for each sign class

## Troubleshooting

### Common Issues:

1. **"No data was loaded" error**:
   - Check that your dataset has the correct structure
   - Ensure each sequence has exactly 20 frames
   - Verify each frame has 225 features

2. **"Expected 225 features" error**:
   - This means the keypoint extraction isn't working correctly
   - Make sure MediaPipe is detecting pose and hand landmarks
   - Check that the camera is working properly

3. **Model loading errors**:
   - Train the model first using `train_model.py`
   - Make sure you have enough training data (at least 10 sequences per sign)

### Performance Tips:

1. **Improve accuracy**:
   - Collect more diverse training data
   - Use data augmentation (scaling, rotation)
   - Adjust the confidence threshold in `main.py`

2. **Speed up training**:
   - Use GPU acceleration if available
   - Reduce the number of epochs if overfitting occurs
   - Use early stopping (already implemented)

## File Descriptions

- `main.py`: Main application for real-time inference
- `train_model.py`: Training script with improved architecture and validation
- `preproccess.py`: Data collection script for creating training sequences
- `my_functions.py`: Core utility functions for keypoint extraction
- `aaa.py`: Video augmentation for creating scaled training data
- `bbb.py`: Visualization tool for comparing different data scales
- `requirements.txt`: Python package dependencies
- `README.md`: This documentation file
