import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def byte_to_binary_array(byte_value):
    """Converts a byte (0-255) into a binary array."""
    return np.array([int(x) for x in f"{byte_value:08b}"])


def detailed_prediction_for_byte(byte_value):
    """Makes detailed predictions for each bit of a given byte value."""
    byte_array = byte_to_binary_array(byte_value)
    print(f"\nStarting detailed prediction for byte value: {byte_value} - Binary: {byte_array}")

    for bit in range(8):
        model_path = f"bit_model_{bit}.keras"
        model = load_model(model_path)
        print(f"\nNow predicting bit {bit} (model loaded from {model_path}):")
        # Prepare the input in the shape model expects
        input_data = byte_array.reshape(1, -1)
        prediction_probability = model.predict(input_data)[0, 0]
        predicted_bit = int(round(prediction_probability))
        print(f"  Input to model: {byte_array}")
        print(f"  Prediction probability for bit {bit}: {prediction_probability:.4f}")
        print(f"  Predicted bit {bit}: {predicted_bit} (rounded from {prediction_probability:.4f})")


if __name__ == "__main__":
    byte_value_to_test = 42  # You can change this value to test other bytes
    detailed_prediction_for_byte(byte_value_to_test)
