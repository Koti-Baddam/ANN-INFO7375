import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Activation Functions
class Activation:
    @staticmethod
    def relu(x): return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x): return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x): return np.tanh(x)

    @staticmethod
    def tanh_derivative(x): return 1 - np.tanh(x) ** 2

    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x): return x * (1 - x)

# Neural Network Layer
class Layer:
    def __init__(self, input_size, output_size, activation='tanh'):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.zeros((output_size, 1))
        self.activation_func = getattr(Activation, activation)
        self.activation_derivative = getattr(Activation, activation + "_derivative")
        self.m = np.zeros_like(self.weights)  # Momentum for Adam
        self.v = np.zeros_like(self.weights)  # RMSProp for Adam

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.biases
        self.output = self.activation_func(self.z)
        return self.output

    def backward(self, d_output, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        d_activation = self.activation_derivative(self.output) * d_output
        d_weights = np.dot(d_activation, self.inputs.T)
        d_biases = np.sum(d_activation, axis=1, keepdims=True)

        # Adam Optimizer
        self.m = beta1 * self.m + (1 - beta1) * d_weights
        self.v = beta2 * self.v + (1 - beta2) * (d_weights ** 2)
        m_hat = self.m / (1 - beta1 ** t)
        v_hat = self.v / (1 - beta2 ** t)
        self.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        self.biases -= learning_rate * d_biases

        return np.dot(self.weights.T, d_activation)  # Return error for next layer

# Neural Network Model
class NeuralNetwork:
    def __init__(self, layer_sizes, activations):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activations[i]) for i in range(len(layer_sizes) - 1)]
        self.loss_history = []

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, y_true, y_pred, learning_rate, t):
        d_loss = y_pred - y_true
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss, learning_rate, t)

    def train(self, X, y, epochs, learning_rate):
        print("\nTraining the Neural Network...\n")
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            self.backward(y, y_pred, learning_rate, epoch)
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.5f}")

        print("\nTraining Completed!")
        self.visualize_loss()

    def visualize_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label="Loss Over Time", color='blue')
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.title("Neural Network Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, X):
        return self.forward(X)

    def visualize_heatmap(self, X):
        predictions = self.predict(X)
        predictions_rounded = np.round(predictions)  # Convert predictions to binary

        plt.figure(figsize=(6, 4))
        sns.heatmap(predictions_rounded, annot=True, cmap="Blues", xticklabels=["0,0", "0,1", "1,0", "1,1"], yticklabels=["Output"])
        plt.xlabel("Input Cases (XOR)")
        plt.ylabel("Predicted Output")
        plt.title("Neural Network Predictions Heatmap")
        plt.show()

    def visualize_decision_boundary(self, X):
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        grid_points = np.c_[xx.ravel(), yy.ravel()].T
        predictions = self.predict(grid_points)
        predictions = np.round(predictions).reshape(xx.shape)

        plt.figure(figsize=(6, 4))
        plt.contourf(xx, yy, predictions, alpha=0.7, cmap="coolwarm")
        plt.scatter(X[0, :], X[1, :], c=y_train[0], cmap="coolwarm", edgecolors="k", marker="o", s=100)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Decision Boundary of Neural Network")
        plt.show()

# Running the Model
if __name__ == "__main__":
    print("\nInitializing Neural Network...")

    # XOR Dataset
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y_train = np.array([[0, 1, 1, 0]])

    # Define a 3-layer neural network (Tanh + Sigmoid in Output)
    model = NeuralNetwork(layer_sizes=[2, 6, 1], activations=['tanh', 'sigmoid'])

    print("\nTraining Starts...\n")
    model.train(X_train, y_train, epochs=1000, learning_rate=0.01)

    print("\nMaking Predictions on XOR Dataset")
    predictions = model.predict(X_train)

    for i, (x, pred) in enumerate(zip(X_train.T, predictions.T)):
        print(f"Input: {x} → Prediction: {pred[0]:.4f}")

    # Heatmap Visualization
    model.visualize_heatmap(X_train)

    # Decision Boundary Visualization
    model.visualize_decision_boundary(X_train)

    print("\nNeural Network Execution Finished!")
