"""
Class 3 — Backpropagation, from scratch.

Tiny network:   x --(w1, b1)--> h --(w2, b2)--> yhat --> Loss (vs target y)

Every number here matches the worked example in notes.md, so you can
run this file and follow along line by line.

No autograd, no PyTorch. Just the chain rule, written out by hand.
"""

import math

# ---------------------------------------------------------
# 1. The network's parameters (what we're trying to learn)
# ---------------------------------------------------------
w1, b1 = 0.5, 0.0     # weight + bias, input -> hidden neuron
w2, b2 = 1.0, 0.0     # weight + bias, hidden -> output neuron

x = 2.0               # one training input
y = 1.0               # the target we wanted


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------
# 2. FORWARD PASS — one straight walk left to right
# ---------------------------------------------------------
h_in = w1 * x + b1
h = sigmoid(h_in)

o_in = w2 * h + b2
yhat = sigmoid(o_in)

loss = (yhat - y) ** 2

print("=== forward pass ===")
print(f"h_in = {h_in:.4f}   h = {h:.4f}")
print(f"o_in = {o_in:.4f}   yhat = {yhat:.4f}")
print(f"loss = {loss:.4f}")


# ---------------------------------------------------------
# 3. BACKWARD PASS — one walk right to left, chain rule only
#    Each line reuses the line above it. Nothing is recomputed
#    from a fresh forward pass.
# ---------------------------------------------------------
dL_dyhat = 2 * (yhat - y)                 # dL/dyhat
dyhat_doin = yhat * (1 - yhat)            # sigmoid'(o_in)
dL_doin = dL_dyhat * dyhat_doin           # dL/do_in  <- reused twice below

dL_dw2 = dL_doin * h                      # dL/do_in * do_in/dw2  (do_in/dw2 = h)
dL_db2 = dL_doin * 1.0                    # do_in/db2 = 1

dL_dh = dL_doin * w2                      # dL/do_in * do_in/dh  (do_in/dh = w2)
dh_dhin = h * (1 - h)                     # sigmoid'(h_in)
dL_dhin = dL_dh * dh_dhin                 # dL/dh_in  <- reused twice below

dL_dw1 = dL_dhin * x                      # dL/dh_in * dh_in/dw1 (dh_in/dw1 = x)
dL_db1 = dL_dhin * 1.0                    # dh_in/db1 = 1

print("\n=== backward pass (analytic gradients) ===")
print(f"dL/dw2 = {dL_dw2:.5f}   dL/db2 = {dL_db2:.5f}")
print(f"dL/dw1 = {dL_dw1:.5f}   dL/db1 = {dL_db1:.5f}")


# ---------------------------------------------------------
# 4. SANITY CHECK — the old "tiny-wiggle" trick from Class 2,
#    used ONLY to prove the backward-pass math above is correct.
#    (This is why we don't use wiggling to actually train: it
#    needs one full forward pass PER parameter, per check.)
# ---------------------------------------------------------
def loss_with(w1, b1, w2, b2):
    h_ = sigmoid(w1 * x + b1)
    yhat_ = sigmoid(w2 * h_ + b2)
    return (yhat_ - y) ** 2


def numerical_gradient(param_name, epsilon=1e-5):
    params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    plus = dict(params)
    minus = dict(params)
    plus[param_name] += epsilon
    minus[param_name] -= epsilon
    loss_plus = loss_with(**plus)
    loss_minus = loss_with(**minus)
    return (loss_plus - loss_minus) / (2 * epsilon)


print("\n=== wiggle-trick check (should ~match the analytic values above) ===")
for name, analytic in [("w1", dL_dw1), ("b1", dL_db1), ("w2", dL_dw2), ("b2", dL_db2)]:
    numeric = numerical_gradient(name)
    print(f"{name}: analytic = {analytic:.5f}   wiggled = {numeric:.5f}")
