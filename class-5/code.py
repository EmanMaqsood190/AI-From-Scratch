import numpy as np
# ------------------------------------------
# 1. Training Data
# ------------------------------------------

# Inputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# Correct XOR outputs
y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# ------------------------------------------
# 2. Activation Function
# ------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ------------------------------------------
# 3. Initialize Weights and Biases
# ------------------------------------------

np.random.seed(1)

# Input → Hidden Layer
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))

# Hidden → Output Layer
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.5


# ------------------------------------------
# 4. Training
# ------------------------------------------

for epoch in range(50000):

    # ===== FORWARD PROPAGATION =====

    # Input → Hidden Layer
    z1 = X @ W1 + b1
    h = sigmoid(z1)

    # Hidden Layer → Output
    z2 = h @ W2 + b2
    prediction = sigmoid(z2)


    # ===== CALCULATE LOSS =====

    loss = np.mean((prediction - y) ** 2)


    # ===== BACKPROPAGATION =====

    # Output layer gradient
    d_prediction = 2 * (prediction - y) / len(X)
    dz2 = d_prediction * prediction * (1 - prediction)

    dW2 = h.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Hidden layer gradient
    dh = dz2 @ W2.T
    dz1 = dh * h * (1 - h)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)


    # ===== GRADIENT DESCENT =====

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# ------------------------------------------
# 5. Final Predictions
# ------------------------------------------

print("XOR Predictions:")
print("----------------")

for i in range(len(X)):

    result = prediction[i][0]

    print(
        f"{int(X[i][0])} XOR {int(X[i][1])} "
        f"= {result:.3f}"
    )
