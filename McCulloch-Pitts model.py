import numpy as np

class MCPNeuron:
    def __init__(self, weights, threshold):
        self.weights = np.array(weights)
        self.threshold = threshold

    def activate(self, inputs):
        return 1 if np.dot(self.weights, inputs) >= self.threshold else 0


def simulate_mcp_neuron():
    print("McCulloch-Pitts Neuron Simulation")

    num_inputs = int(input("Number of inputs: "))
    weights = [float(input(f"Weight {i + 1}: ")) for i in range(num_inputs)]
    threshold = float(input("Threshold value: "))

    neuron = MCPNeuron(weights, threshold)
    input_combinations = [list(map(int, bin(i)[2:].zfill(num_inputs))) for i in range(2 ** num_inputs)]

    print("\nInputs | Output")
    print("-" * (num_inputs * 3 + 10))
    for inputs in input_combinations:
        print(f"{inputs} -> {neuron.activate(inputs)}")


if __name__ == "__main__":
    simulate_mcp_neuron()
