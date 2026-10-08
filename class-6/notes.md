# 📘 Class 6 | Neural Network Training

## 🔄 Quick Recall — Class 5
In Class 5, we built a 2-layer neural network to solve the XOR problem. Since a single neuron cannot solve XOR because it is not linearly separable, we added a hidden layer between the input and output layers. We connected forward propagation, loss, backpropagation, gradients, and gradient descent to train the network and adjust its weights and biases so that it could learn the XOR pattern.

# 🔹 1. What Is Neural Network Training?



Now in Class 6, we learn **how to train the network repeatedly so that it becomes better at making predictions**.

Training means:

> Repeatedly showing examples to the neural network, calculating its error, and adjusting its weights and biases to reduce that error.

The basic training process is:

```text
Input Data
    ↓
Forward Propagation
    ↓
Prediction
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Calculate Gradients
    ↓
Update Weights
    ↓
Repeat
```

There are 3 important concepts:

1. Epoch
2. Batch Size
3. Learning Rate
---

# 🔹 2. What Is an Epoch?

An **epoch** means that the neural network has gone through the **entire training dataset once**.

For example, our XOR dataset contains four examples:

```text
0, 0 → 0
0, 1 → 1
1, 0 → 1
1, 1 → 0
```

If the network trains on all four examples once:

```text
1 Epoch
```

If it goes through the complete dataset 10 times:

```text
10 Epochs
```

So:

```text
1 Epoch = One complete pass through the training dataset
```


---

# 🔹 3. What Is Batch size?

A **batch** is a smaller group of training examples.

The **batch size** tells us how many examples the network processes.

For example:

```text
Dataset = 100 examples
Batch Size = 10
```

The dataset can be divided into:

```text
Batch 1 → 10 examples
Batch 2 → 10 examples
Batch 3 → 10 examples
...
Batch 10 → 10 examples
```

After all 10 batches have been processed:

```text
1 Epoch is complete
```

So:

```text
Batch Size = Number of training examples processed in one batch
```

# 🔹 4. What Is the Learning Rate?

The **learning rate** controls how much the weights and biases change during training.

We saw the basic update rule in Class 5:

```text
new weight = old weight - learning rate × gradient
```

For example:

```text
Old Weight = 0.5
Gradient = 0.2
Learning Rate = 0.1
```

Then:

```text
new weight = 0.5 - (0.1 × 0.2)

new weight = 0.48
```

The weight changes from:

```text
0.5 → 0.48
```


---

## 🔹 Small Learning Rate

Suppose:

```text
Learning Rate = 0.001
```

The weight changes very slowly.

```text
Small Learning Rate
        ↓
Small Updates
        ↓
Slow Learning
```



---

## 🔹 Large Learning Rate

Suppose the learning rate is very large.

The weights may change too much at each step.

```text
Large Learning Rate
        ↓
Large Updates
        ↓
May Jump Around
        ↓
Training Can Become Unstable
```

Therefore, we need a suitable learning rate.

---


# 🔹 5. How They Work Together

Suppose we have:

```text
Dataset = 100 examples
Batch Size = 10
Epochs = 5
Learning Rate = 0.01
```

First, we divide the dataset:

```text
100 examples ÷ 10
= 10 batches
```

Therefore:

```text
1 Epoch = 10 Batches
```

For 5 epochs:

```text
5 × 10 = 50 batches
```

If the network updates its weights after every batch, there will be:

```text
50 weight updates
```

during the 5 epochs.

The learning rate `0.01` controls the size of each update.

---

# 🔹 6. Complete Neural Network Training Process

Now we can connect everything together:

```text
                 DATASET
                    ↓
                 Batches
                    ↓
          Forward Propagation
                    ↓
                Prediction
                    ↓
                   Loss
                    ↓
            Backpropagation
                    ↓
                Gradients
                    ↓
             Weight Updates
                    ↑
              Learning Rate
                    ↓
               Next Batch
                    ↓
          All Batches Finished
                    ↓
                1 Epoch
                    ↓
              Next Epoch
                    ↓
                 Repeat
```

The network repeats this process many times.

The goal is to gradually reduce the loss and improve the predictions.

---

# 🧠 Key Takeaways

- **Neural network training** means repeatedly adjusting the network's parameters to reduce the loss.
- An **epoch** is one complete pass through the training dataset.
- A **batch** is a smaller group of training examples.
- **Batch size** determines how many examples are processed in one batch.
- **Learning rate** controls the size of weight and bias updates.
- A small learning rate makes learning slower.
- A very large learning rate can make training unstable.
- The network uses **forward propagation** to make predictions.
- **Loss** measures how wrong the predictions are.
- **Backpropagation** calculates gradients.
- **Gradient descent** uses the gradients to update weights and biases.
- Repeating this process allows the network to gradually learn the desired pattern.

---

# 🔜 Up Next — Class 7
In  Class 7, we will evaluate and test the trained neural network by checking its predictions, accuracy, and loss.

