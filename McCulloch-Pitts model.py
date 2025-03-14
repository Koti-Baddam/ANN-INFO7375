# McCulloch and Pitts Model Simulation

def mcculloch_pitts_neuron(inputs, weights, threshold):
    """
    Simulates a McCulloch-Pitts Neuron.
    Args:
        inputs (list): List of binary inputs (0 or 1).
        weights (list): Corresponding weights for each input.
        threshold (int): Activation threshold.

    Returns:
        int: Output (1 if activated, 0 otherwise).
    """
    # Calculate the weighted sum of inputs
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))

    # Apply the activation function
    if weighted_sum >= threshold:
        return 1
    else:
        return 0


# Example inputs
inputs = [1, 0, 1]  # Binary inputs
weights = [2, 1, 3]  # Weights for each input
threshold = 4  # Activation threshold

# Simulate the neuron
output = mcculloch_pitts_neuron(inputs, weights, threshold)

print(f"Inputs: {inputs}")
print(f"Weights: {weights}")
print(f"Threshold: {threshold}")
print(f"Output: {output}")