from data_preparation import load_images
from perceptron_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Load test data
X_test, Y_test = load_images("test_digits")

# Load the trained model
model_data = np.load("perceptron_model.npz")
perceptron = Perceptron(input_size=20 * 20, output_size=10)
perceptron.weights = model_data["weights"]
perceptron.bias = model_data["bias"]

print("Testing the Perceptron on test data...")

# Predict the labels for the test data
predictions = perceptron.predict(X_test)
predicted_labels = [np.argmax(pred) for pred in predictions.T]

# Calculate accuracy
accuracy = accuracy_score(Y_test, predicted_labels)
print(f"Test Accuracy: {accuracy:.2%}")

# Generate confusion matrix
conf_matrix = confusion_matrix(Y_test, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Display individual predictions
for i, (true_label, predicted_label) in enumerate(zip(Y_test, predicted_labels), start=1):
    print(f"Test Image {i}: True Label -> {true_label}, Predicted Label -> {predicted_label}")
