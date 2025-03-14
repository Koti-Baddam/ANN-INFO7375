import numpy as np

class Perceptron:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, X):
        self.Z = np.dot(self.weights, X) + self.bias
        self.A = self.sigmoid(self.Z)
        return self.A

    def backward(self, X, Y, learning_rate):
        m = X.shape[1]
        dZ = self.A - Y
        dW = (1 / m) * np.dot(dZ, X.T)
        dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * dB

    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            A = self.forward(X)
            loss = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))
            self.backward(X, Y, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        A = self.forward(X)
        return (A > 0.5).astype(int)
