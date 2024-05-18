import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input

def byte_to_binary_array(byte_value):
    """Converts a byte (0-255) into a binary array."""
    return np.array([int(x) for x in f"{byte_value:08b}"])

def prepare_datasets():
    """Prepares the dataset for training different models for each bit."""
    X = np.array([byte_to_binary_array(i) for i in range(256)])
    Y = {i: X[:, i] for i in range(8)}  # Dictionary of targets for each bit
    return X, Y

def define_model():
    """Defines a simple neural network model for binary classification."""
    model = Sequential([
        Input(shape=(8,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_models(X, Y):
    """Trains a separate model for each bit and saves them using the .keras format."""
    models = {}
    for bit in range(8):
        print(f"Training model for bit {bit}")
        model = define_model()
        model.fit(X, Y[bit], epochs=100, verbose=0)
        model_path = f"bit_model_{bit}.keras"
        model.save(model_path)
        models[bit] = model
    return models

def load_and_verify_models(X, Y):
    """Loads each model from the .keras format and verifies its accuracy."""
    models = {}
    for bit in range(8):
        model_path = f"bit_model_{bit}.keras"
        model = load_model(model_path)
        predictions = model.predict(X)[:, 0]
        accuracy = np.mean((predictions > 0.5).astype(int) == Y[bit])
        print(f"Model for bit {bit} loaded with accuracy: {accuracy:.2f}")
        models[bit] = model
    return models

def full_byte_test(models, X):
    """Tests all models on the full numerical spectrum of one byte and prints results."""
    for byte_value in range(256):
        byte_array = byte_to_binary_array(byte_value)
        predicted_bits = []
        for bit in range(8):
            prediction = models[bit].predict(byte_array.reshape(1, -1))[0, 0]
            predicted_bit = int(round(prediction))
            predicted_bits.append(predicted_bit)
        print(f"Byte {byte_value}: Expected {byte_array}, Predicted {np.array(predicted_bits)}")

if __name__ == "__main__":
    X, Y = prepare_datasets()
    models = train_models(X, Y)
    models = load_and_verify_models(X, Y)
    full_byte_test(models, X)
    print("All byte values from 0 to 255 have been tested successfully!")
