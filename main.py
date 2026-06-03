import numpy as np

# =====================================
# DATA
# =====================================

input_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

current_input_index = 0

# =====================================
# HYPERPARAMETERS
# =====================================

learning_rate = 0.1

# =====================================
# ACTIVATIONS
# =====================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# =====================================
# WEIGHTS (4 hidden neurons)
# =====================================

# 2 -> 4
W1 = np.random.uniform(-0.5, 0.5, (4, 2))
b1 = np.random.uniform(-0.5, 0.5, (4, 1))

# 4 -> 1
W2 = np.random.uniform(-0.5, 0.5, (1, 4))
b2 = np.random.uniform(-0.5, 0.5, (1, 1))


# =====================================
# TRAINING LOOP
# =====================================

for epoch in range(10000):

    # ==============================
    # INPUT
    # ==============================
    x = np.array(input_data[current_input_index]).reshape(2, 1)

    current_input_index = (current_input_index + 1) % len(input_data)

    # ==============================
    # TARGET (XOR)
    # ==============================
    x_flat = x.flatten()

    if np.array_equal(x_flat, [0, 0]):
        y_target = 0
    elif np.array_equal(x_flat, [0, 1]):
        y_target = 1
    elif np.array_equal(x_flat, [1, 0]):
        y_target = 1
    else:
        y_target = 0

    y_target = np.array([[y_target]])

    # ==============================
    # FORWARD
    # ==============================
    z1 = W1 @ x + b1
    a1 = sigmoid(z1)

    z2 = W2 @ a1 + b2
    y_pred = sigmoid(z2)

    loss = (y_pred - y_target) ** 2

    # ==============================
    # BACKPROP
    # ==============================
    dL_dy = 2 * (y_pred - y_target)
    dz2 = dL_dy * sigmoid_derivative(z2)

    dW2 = dz2 @ a1.T
    db2 = dz2

    da1 = W2.T @ dz2
    dz1 = da1 * sigmoid_derivative(z1)

    dW1 = dz1 @ x.T
    db1 = dz1

    # ==============================
    # UPDATE
    # ==============================
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # ==============================
    # BEAUTIFUL LOGGING
    # ==============================
    if epoch % 200 == 0:
        print("\n" + "=" * 60)
        print(f"EPOCH: {epoch}")
        print("-" * 60)

        print(f"input      : {x_flat}")
        print(f"target     : {y_target.item():.0f}")
        print(f"prediction : {y_pred.item():.4f}")
        print(f"loss       : {loss.item():.6f}")

        print("\nhidden activations (a1):")
        print(np.round(a1.flatten(), 4))

        print("\nweights W1:")
        print(np.round(W1, 4))

        print("\nweights W2:")
        print(np.round(W2, 4))

        print("=" * 60)