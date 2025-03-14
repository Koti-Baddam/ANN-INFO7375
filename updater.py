"update over main.py for better model performance "
import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
class Activation:
    """Handles activation functions and their derivatives."""

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

# Neuron Class
class Neuron:
    """Defines a single neuron with weights and bias."""

    def __init__(self, input_size, activation="sigmoid"):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.activation_func = Activation.relu if activation == "relu" else Activation.sigmoid
        self.activation_derivative = Activation.relu_derivative if activation == "relu" else Activation.sigmoid_derivative

    def compute(self, inputs):
        """Computes the neuron's output."""
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation_func(self.z)
        return self.output

    def update(self, error, learning_rate):
        """Updates weights and bias."""
        gradient = self.activation_derivative(self.output)
        self.weights -= learning_rate * error * gradient * self.inputs
        self.bias -= learning_rate * error * gradient

# Layer Class
class Layer:
    """Represents a collection of neurons working together."""

    def __init__(self, num_neurons, num_inputs, activation="sigmoid"):
        self.neurons = [Neuron(num_inputs, activation) for _ in range(num_neurons)]

    def activate(self, inputs):
        """Pass inputs through all neurons in the layer."""
        return np.array([neuron.compute(inputs) for neuron in self.neurons])

    def adjust(self, errors, learning_rate):
        """Adjusts neurons' weights using backpropagation."""
        for i, neuron in enumerate(self.neurons):
            neuron.update(errors[i], learning_rate)

# Parameters Class
class Parameters:
    """Manages the neural network layers."""

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Layer(hidden_size, input_size, activation="relu")  # Using ReLU
        self.output_layer = Layer(output_size, hidden_size, activation="sigmoid")  # Sigmoid in output

# Loss Function Class
class LossFunction:
    """Computes loss and its derivative."""

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        return -(y_true - y_pred)

# Forward Propagation
class ForwardProp:
    """Handles the forward pass through the network."""

    @staticmethod
    def execute(model, inputs):
        hidden_output = model.hidden_layer.activate(inputs)
        final_output = model.output_layer.activate(hidden_output)
        return hidden_output, final_output

# Backward Propagation
class BackProp:
    """Handles weight updates using backpropagation."""

    @staticmethod
    def execute(model, inputs, hidden_output, output, y_true, learning_rate):
        output_errors = LossFunction.mse_derivative(y_true, output)
        model.output_layer.adjust(output_errors, learning_rate)

        hidden_errors = np.dot(output_errors, np.array([neuron.weights for neuron in model.output_layer.neurons]))
        model.hidden_layer.adjust(hidden_errors, learning_rate)

# Gradient Descent (with Mini-Batch)
class GradDescent:
    """Optimizes weights using mini-batch gradient descent."""

    @staticmethod
    def update(model, inputs_batch, y_true_batch, learning_rate):
        total_loss = 0
        batch_size = len(inputs_batch)

        for i in range(batch_size):
            hidden_output, output = ForwardProp.execute(model, inputs_batch[i])
            BackProp.execute(model, inputs_batch[i], hidden_output, output, y_true_batch[i], learning_rate)
            total_loss += LossFunction.mse(y_true_batch[i], output)

        return total_loss / batch_size  # Average loss over mini-batch

# Neural Network Model
class Model:
    """Defines the complete neural network."""

    def __init__(self, input_size, hidden_size, output_size):
        self.params = Parameters(input_size, hidden_size, output_size)

    def predict(self, inputs):
        """Generates predictions from the model."""
        _, output = ForwardProp.execute(self.params, inputs)
        return output

# Training with Mini-Batch
class Training:
    """Handles model training and visualization."""

    def __init__(self, model, learning_rate=0.1, epochs=1000, batch_size=2):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history = []

    def run(self, X_train, y_train):
        """Executes training with mini-batches."""
        num_samples = len(X_train)

        for epoch in range(self.epochs):
            total_loss = 0
            indices = np.random.permutation(num_samples)  # Shuffle data

            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                inputs_batch = X_train[batch_indices]
                y_true_batch = y_train[batch_indices]

                total_loss += GradDescent.update(self.model.params, inputs_batch, y_true_batch, self.learning_rate)

            avg_loss = total_loss / (num_samples / self.batch_size)
            self.loss_history.append(avg_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.5f}")

    def visualize_loss(self):
        """Plots the loss curve."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label="Loss Over Time", color='blue')
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.title("Training Loss Curve (Mini-Batch)")
        plt.legend()
        plt.grid(True)
        plt.show()

# Running the Model
if __name__ == "__main__":
    # XOR Dataset
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    # Initialize Model with ReLU and Mini-Batch
    model = Model(input_size=2, hidden_size=5, output_size=1)  # Increased Hidden Size
    trainer = Training(model, learning_rate=0.05, epochs=1000, batch_size=2)  # Mini-Batch
    trainer.run(X_train, y_train)
    trainer.visualize_loss()
