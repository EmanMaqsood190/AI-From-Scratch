"""
Class 4 | Activation Functions - Simple Version

What this code does, in plain words:
1. Builds the three gates: sigmoid, tanh, relu (and leaky relu)
2. Builds their "slopes" (used for backprop / learning)
3. Runs the same neuron through all three gates and shows what happens
4. Shows WHY sigmoid gets stuck (vanishing gradient) but ReLU doesn't
"""

import math


# ---------------------------------------------------------
# STEP 1: The gates
# ---------------------------------------------------------

def sigmoid(z):
    # Turns any number into something between 0 and 1
    return 1 / (1 + math.exp(-z))


def sigmoid_slope(z):
    # How much sigmoid reacts to change at this point
    # Biggest possible value is 0.25
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    # Turns any number into something between -1 and 1
    return math.tanh(z)


def tanh_slope(z):
    # Biggest possible value is 1.0
    t = tanh(z)
    return 1 - t ** 2


def relu(z):
    # Simple rule: negative becomes 0, positive stays the same
    return max(0.0, z)


def relu_slope(z):
    # 1 if the number was positive, 0 if it was negative
    return 1.0 if z > 0 else 0.0


def leaky_relu(z, leak=0.01):
    # Same as relu, but keeps a tiny bit alive for negative numbers
    return z if z > 0 else leak * z


def leaky_relu_slope(z, leak=0.01):
    return 1.0 if z > 0 else leak


# ---------------------------------------------------------
# STEP 2: The neuron from Class 1 (unchanged)
# ---------------------------------------------------------

def neuron_output(x, w, b):
    # z = (weights times inputs, added up) + bias
    return sum(wi * xi for wi, xi in zip(w, x)) + b


# ---------------------------------------------------------
# STEP 3: Same neuron, three gates (the example from the notes)
# ---------------------------------------------------------

def run_example():
    x = [2, 3]
    w = [0.5, -1]
    b = 1

    z = neuron_output(x, w, b)
    print("=== Same neuron, three gates ===")
    print(f"inputs = {x}, weights = {w}, bias = {b}")
    print(f"z (before any gate) = {z}\n")

    print(f"{'Gate':<12}{'Output':<12}{'Slope':<12}")
    print("-" * 36)
    print(f"{'ReLU':<12}{relu(z):<12.3f}{relu_slope(z):<12.3f}")
    print(f"{'Sigmoid':<12}{sigmoid(z):<12.3f}{sigmoid_slope(z):<12.3f}")
    print(f"{'Tanh':<12}{tanh(z):<12.3f}{tanh_slope(z):<12.3f}")

    print("\n--- Now try with bias = 4 instead of 1 ---")
    b2 = 4
    z2 = neuron_output(x, w, b2)
    print(f"z = {z2}\n")
    print(f"{'Gate':<12}{'Output':<12}{'Slope':<12}")
    print("-" * 36)
    print(f"{'ReLU':<12}{relu(z2):<12.3f}{relu_slope(z2):<12.3f}")
    print(f"{'Sigmoid':<12}{sigmoid(z2):<12.3f}{sigmoid_slope(z2):<12.3f}")
    print(f"{'Tanh':<12}{tanh(z2):<12.3f}{tanh_slope(z2):<12.3f}")
    print()


# ---------------------------------------------------------
# STEP 4: Why sigmoid gets "stuck" but ReLU doesn't
# ---------------------------------------------------------

def why_relu_wins(num_stations=10):
    print(f"=== What happens after passing through {num_stations} stations ===\n")

    # Best case slope for each gate, multiplied together many times
    sigmoid_total = 0.25 ** num_stations
    relu_total = 1.0 ** num_stations

    print(f"Sigmoid: 0.25 multiplied {num_stations} times = {sigmoid_total:.10f}")
    print("  -> the learning signal is almost completely gone")
    print()
    print(f"ReLU: 1 multiplied {num_stations} times = {relu_total:.10f}")
    print("  -> the learning signal survives the whole trip")
    print()
    print("This is why deep networks made of sigmoid gates learn very slowly,")
    print("while deep networks made of ReLU gates learn much faster.\n")


# ---------------------------------------------------------
if __name__ == "__main__":
    run_example()
    why_relu_wins()
