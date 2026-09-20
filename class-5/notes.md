# 📘 Class 5 | Building a 2-Layer Neural Network — Solving XOR

In the previous classes, we learned how a single neuron works, how to calculate loss, how backpropagation sends the error backward, how gradient descent updates weights, and why activation functions are needed.

Now we'll connect all these concepts together and build a small neural network that can solve the **XOR problem**.

---

## 🔹 1. The XOR Problem

XOR gives `1` when the two inputs are different and `0` when they are the same.

| Input 1 | Input 2 | XOR Output |
|--------:|--------:|-----------:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

So:

```text
0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0
```

---

## 🔹 2. Why Can't One Neuron Solve XOR?

A single neuron can learn simple **linear patterns**.

XOR, however, is **not linearly separable**.

This means we cannot draw one straight line that separates all the `0` outputs from all the `1` outputs.

So:

```text
Single Neuron
Input → Output
     ❌
Cannot solve XOR
```

We need something more powerful.

---

## 🔹 3. Adding a Hidden Layer

Instead of using only one neuron, we add a **hidden layer**.

```text
Input Layer → Hidden Layer → Output Layer
```

Our small network can have:

- 2 input neurons
- 2 hidden neurons
- 1 output neuron

```text
       Hidden Layer
      ┌───────────┐
x₁ ──→│    h₁     │───┐
      └───────────┘   │
                      ├──→ Output
      ┌───────────┐   │
x₂ ──→│    h₂     │───┘
      └───────────┘
```

The input layer is not counted as a trainable layer.

So this is called a **2-layer neural network**:

```text
Input → Layer 1 (Hidden) → Layer 2 (Output)
```

---

## 🔹 4. What Does a Hidden Neuron Do?

A hidden neuron works just like the neuron we learned in Class 1.

It calculates:

```text
z = w₁x₁ + w₂x₂ + b
```

Then applies an activation function:

```text
h = f(z)
```

Each hidden neuron can learn a different useful pattern from the inputs.

The output layer then combines these patterns to make the final prediction.

---

## 🔹 5. Why Do We Need Activation Functions?

Without an activation function, every layer would behave like a linear equation.

Adding more linear layers would still give us a linear function.

Activation functions introduce **non-linearity**.

This allows the network to learn more complicated patterns such as XOR.

```text
Weighted Sum
     ↓
Activation Function
     ↓
Neuron Output
```

---

## 🔹 6. Forward Propagation

When information moves from the input toward the output, it is called **forward propagation**.

```text
Input
  ↓
Hidden Layer
  ↓
Output Layer
  ↓
Prediction
```

For the hidden layer:

```text
z₁ = w₁x₁ + w₂x₂ + b
h₁ = f(z₁)
```

Similarly, the second hidden neuron produces `h₂`.

Then the output neuron uses `h₁` and `h₂`:

```text
z = v₁h₁ + v₂h₂ + b
ŷ = f(z)
```

Here `ŷ` means the network's prediction.

---

## 🔹 7. Prediction → Loss

After getting a prediction, we compare it with the real answer.

For example:

```text
Actual = 1
Prediction = 0.3
```

The difference is measured using a **loss function**.

```text
Prediction
    ↓
  Loss
```

A larger error means a larger loss.

---

## 🔹 8. Backpropagation

Now the network needs to know:

> Which weights caused the error, and how should they change?

That's where **backpropagation** comes in.

It sends the error information backward through the network and calculates the **gradients**.

```text
Loss
 ↓
Output Layer
 ↓
Hidden Layer
```

Each weight gets information about how it contributed to the loss.

---

## 🔹 9. Gradient Descent

Once we have the gradients, **gradient descent** updates the weights and biases.

```text
Old Weight → Gradient → New Weight
```

The basic update rule is:

```text
new weight = old weight - learning rate × gradient
```

We keep repeating this process so that the loss becomes smaller.

---

## 🔹 10. The Complete Learning Process

Now all the concepts from Classes 1–5 connect together:

```text
              FORWARD
Input → Hidden Layer → Output
                         ↓
                    Prediction
                         ↓
                        Loss
                         ↓
              BACKPROPAGATION
                         ↓
                     Gradients
                         ↓
               GRADIENT DESCENT
                         ↓
                 Update Weights
                         ↓
                      Repeat
```

The network repeats this process many times while learning from the XOR examples.

---

## 🔹 11. What Eventually Happens?

Initially, the network's weights are not correct, so its predictions are poor.

After many training steps, the weights are adjusted and the network learns the XOR pattern:

```text
0, 0 → 0
0, 1 → 1
1, 0 → 1
1, 1 → 0
```

The goal is not to manually tell the network these answers.

Instead, the network **learns the required weights through training**.

---

# 🧠 Key Takeaways

- A single neuron cannot solve XOR because XOR is not linearly separable.
- A hidden layer gives the network more ability to learn complex patterns.
- A hidden neuron performs a weighted sum, bias addition, and activation.
- **Forward propagation** produces the prediction.
- **Loss** tells us how wrong the prediction is.
- **Backpropagation** calculates how each parameter contributed to the error.
- **Gradients** tell us the direction of change.
- **Gradient descent** updates the weights and biases.
- Repeating this process allows the network to learn XOR.

---
