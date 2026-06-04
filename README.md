# XOR Neural Network from Scratch

For my main project, I wanted to understand how neural networks actually work at the mathematical level instead of treating frameworks as a black box. To build that intuition, I created a small neural network that learns the XOR function from scratch using only NumPy.

## What is XOR?

XOR (Exclusive OR) is a simple logical operation:

| Input A | Input B | Output |
| ------- | ------- | ------ |
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

The XOR problem is a classic benchmark in machine learning because it cannot be solved by a single linear neuron. A neural network needs at least one hidden layer to learn the non-linear relationship between the inputs and outputs.

## Project Goals

The goal of this project was not to achieve high performance but to understand the mathematics behind neural networks, including:

* Forward propagation
* Sigmoid activation functions
* Loss calculation
* Gradient descent
* Backpropagation
* The chain rule

## Implementation

This neural network is implemented entirely with NumPy.

No machine learning frameworks are used:

* No PyTorch
* No TensorFlow
* No Keras

Every step of the learning process is implemented manually, including weight initialization, forward propagation, gradient computation, and parameter updates.

## Network Architecture

```text
Input Layer (2 neurons)
        ↓
Hidden Layer (4 neurons, Sigmoid)
        ↓
Output Layer (1 neuron, Sigmoid)
```

The network learns the XOR truth table by adjusting its weights through backpropagation and gradient descent.

## Why This Project?

Understanding neural networks at the mathematical level makes it much easier to work with larger models and modern frameworks. By implementing every component manually, it becomes clear how information flows through the network and how gradients are used to update the weights during training.


## Dependencies

This project uses only two libraries:

- NumPy
- Matplotlib


![matholib VISUALIZATION](./pictures/Screenshot%202026-06-04%20at%2016.28.06.png)