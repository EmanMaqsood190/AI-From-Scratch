import numpy as np

# XOR Training Data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Weights and Biases
np.random.seed(1)

W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))


# Training Settings
learning_rate = 0.5
epochs = 10000
batch_size = 2


# Training
for epoch in range(epochs):

    # Divide data into batches
    for start in range(0, len(X), batch_size):

        batch_X = X[start:start + batch_size]
        batch_y = y[start:start + batch_size]

        # Forward Propagation
        z1 = batch_X @ W1 + b1
        h = sigmoid(z1)

        z2 = h @ W2 + b2
        prediction = sigmoid(z2)

        # Loss
        loss = np.mean((prediction - batch_y) ** 2)

        # Backpropagation
        error = prediction - batch_y

        dz2 = error * prediction * (1 - prediction)

        dW2 = h.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ W2.T) * h * (1 - h)

        dW1 = batch_X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update Weights and Biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1


# Final Prediction
z1 = X @ W1 + b1
h = sigmoid(z1)

z2 = h @ W2 + b2
prediction = sigmoid(z2)


# Display Results
print("XOR Results:")
print("----------------")

for i in range(len(X)):
    print(
        X[i],
        "=>",
        round(prediction[i][0], 3)
    )
