"""
Convert TensorFlow model to ONNX for DirectML inference
"""

import tensorflow as tf
import tf2onnx
import onnxruntime as ort

def convert_model_to_onnx(model_path, output_path):
    """Convert TensorFlow model to ONNX format"""
    try:
        # Load the TensorFlow model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to ONNX
        spec = (tf.TensorSpec((None, 20, 225), tf.float32, name="input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        
        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"Model converted to ONNX: {output_path}")
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

def create_directml_inference_session(onnx_path):
    """Create DirectML inference session"""
    try:
        # Create session with DirectML provider
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print("DirectML inference session created successfully")
        return session
        
    except Exception as e:
        print(f"DirectML session creation failed: {e}")
        return None

if __name__ == "__main__":
    # Convert model if it exists
    if os.path.exists("my_model.h5"):
        convert_model_to_onnx("my_model.h5", "model.onnx")
        
        # Test DirectML inference
        session = create_directml_inference_session("model.onnx")
        if session:
            print("DirectML inference ready!")
    else:
        print("No model found. Train the model first.")
