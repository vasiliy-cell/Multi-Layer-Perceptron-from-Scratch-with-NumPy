import numpy as np
import matplotlib.pyplot as plt

# =====================================
# DATA
# =====================================
loss_history = []


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

learning_rate = 8.0

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

for epoch in range(20000):

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
    loss_history.append(loss.item())
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
    # LOGGING
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


# =====================================
# VISUALIZATION
# =====================================
print("\nTraining complete. Preparing the plot...")

# Configure a stylish dark background theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# Main line (original noisy loss with high transparency)
epochs_range = range(len(loss_history))
ax.plot(epochs_range, loss_history, color='#17becf', alpha=0.15, label='Original Loss')

# Smoothed line (moving average for a clean trend line)
# Calculating the average every 500 epochs to eliminate SGD noise
window_size = 500
if len(loss_history) > window_size:
    smooth_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
    smooth_epochs = range(window_size - 1, len(loss_history))
    ax.plot(smooth_epochs, smooth_loss, color='#ff7f0e', linewidth=2, label=f'Smoothed Loss (MA {window_size})')

# Grid and axis customization
ax.set_title('Neural Network Training Progress (XOR)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Epoch', fontsize=12, labelpad=10)
ax.set_ylabel('Loss (MSE)', fontsize=12, labelpad=10)
ax.grid(True, linestyle='--', alpha=0.3, color='gray')

# Legend setup
ax.legend(loc='upper right', frameon=True, facecolor='#222222', edgecolor='none')

# Visual badge showing the final Loss value
final_loss = loss_history[-1]
textstr = f'Final Loss: {final_loss:.6f}'
props = dict(boxstyle='round,pad=0.5', facecolor='#222222', alpha=0.8, edgecolor='#ff7f0e')
ax.text(0.05, 0.15, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)

# Automatically adjust layout and render the plot
plt.tight_layout()
plt.show()