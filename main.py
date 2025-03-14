import numpy as np
import matplotlib.pyplot as plt


# Activation Functions
class Activation:
    """Handles activation and its derivative."""

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)


# Neuron Class
class Neuron:
    """Defines a single neuron with weights and bias."""

    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def compute(self, inputs):
        """Computes activation output."""
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.output = Activation.sigmoid(self.z)
        return self.output

    def update(self, error, learning_rate):
        """Updates weights and bias using backpropagation."""
        gradient = Activation.sigmoid_derivative(self.output)
        self.weights -= learning_rate * error * gradient * self.inputs
        self.bias -= learning_rate * error * gradient


# Layer Class
class Layer:
    """Represents a collection of neurons working together."""

    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def activate(self, inputs):
        """Pass inputs through all neurons in the layer."""
        return np.array([neuron.compute(inputs) for neuron in self.neurons])

    def adjust(self, errors, learning_rate):
        """Backpropagate errors through all neurons."""
        for i, neuron in enumerate(self.neurons):
            neuron.update(errors[i], learning_rate)


# Parameters Class
class Parameters:
    """Manages the neural network structure."""

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Layer(hidden_size, input_size)
        self.output_layer = Layer(output_size, hidden_size)


# Loss Function Class
class LossFunction:
    """Computes loss and its gradient."""

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


# Gradient Descent
class GradDescent:
    """Optimizes weights using gradient descent."""

    @staticmethod
    def update(model, inputs, y_true, learning_rate):
        hidden_output, output = ForwardProp.execute(model, inputs)
        BackProp.execute(model, inputs, hidden_output, output, y_true, learning_rate)
        return LossFunction.mse(y_true, output)


# Neural Network Model
class Model:
    """Defines the complete neural network."""

    def __init__(self, input_size, hidden_size, output_size):
        self.params = Parameters(input_size, hidden_size, output_size)

    def predict(self, inputs):
        """Generates predictions from the model."""
        _, output = ForwardProp.execute(self.params, inputs)
        return output


# Training Process
class Training:
    """Handles model training and loss visualization."""

    def __init__(self, model, learning_rate=0.1, epochs=1000):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def run(self, X_train, y_train):
        """Executes training over multiple epochs."""
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(len(X_train)):
                total_loss += GradDescent.update(self.model.params, X_train[i], y_train[i], self.learning_rate)
            avg_loss = total_loss / len(X_train)
            self.loss_history.append(avg_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.5f}")

    def visualize_loss(self):
        """Plots the loss curve."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label="Loss Over Time", color='blue')
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.show()


# Running the Model
if __name__ == "__main__":
    # XOR Dataset
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    # Initialize and Train Model
    model = Model(input_size=2, hidden_size=3, output_size=1)
    trainer = Training(model, learning_rate=0.1, epochs=1000)
    trainer.run(X_train, y_train)
    trainer.visualize_loss()
