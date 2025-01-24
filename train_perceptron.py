from data_preparation import load_images
from perceptron_model import Perceptron
import numpy as np
import os

# Load training data
X_train, Y_train = load_images("train_digits")
Y_train_one_hot = np.eye(10)[Y_train].T  # Convert labels to one-hot encoding

# Display information about the training data
print(f"Training data loaded: {X_train.shape[1]} samples with {X_train.shape[0]} features each.")

# Ask for hyperparameters
learning_rate = float(input("Enter the learning rate (e.g., 0.1): "))
epochs = int(input("Enter the number of epochs (e.g., 100): "))

# Initialize and train the Perceptron
input_size = 20 * 20  # Image size: 20x20 pixels
output_size = 10  # Output classes: digits 0–9
perceptron = Perceptron(input_size, output_size)

print("Training the Perceptron...")
perceptron.train(X_train, Y_train_one_hot, epochs=epochs, learning_rate=learning_rate)

# Save the trained model
model_path = "perceptron_model.npz"
np.savez(model_path, weights=perceptron.weights, bias=perceptron.bias)
print(f"Model saved as '{model_path}'")

# Optional: Verify that the model file was saved correctly
if os.path.exists(model_path):
    print(f"Trained model successfully saved at: {os.path.abspath(model_path)}")
else:
    print("Error: Failed to save the trained model!")
