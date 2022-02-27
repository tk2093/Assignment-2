from Linear_layer import LinearLayer as LL
from tanh import Tanh
from sigmoid import Sigmoid
import numpy as np


def mse(pred, target):
    return np.mean(np.power(target-pred, 2))


def mse_grad(pred, target):
    return 2*(pred-target)/target.size


x_train = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])
y_train = np.array([[0],
                    [1],
                    [1],
                    [0]])
# Forward Pass

# Initializing input hidden Layer
layer1 = LL(2,2)

# Initializing output layer
outputlayer = LL(2,1)

for j in range(101):
    for i in range(len(y_train)):
        # forward pass

        # Input hidden Layer
        a1 = layer1.forward_pass(x_train[i:i+1, :])
        z1 = Tanh().forward_pass(a1)
        # output layer
        a2 = outputlayer.forward_pass(z1.T)
        z2 = Tanh().forward_pass(a2)

        # Backward pass

        # Output layer
        error = mse_grad(z2, y_train[i:i+1, :])
        error = Tanh().backward_pass(error)
        error = outputlayer.backward_pass(error)
        # Input hidden Layer
        error = Tanh().backward_pass(error)
        error = layer1.backward_pass(error)

prediction = []

for i in range(len(y_train)):
    pred = x_train[i:i+1, :]
    pred = layer1.forward_pass(pred)
    pred = outputlayer.forward_pass(pred.T)
    prediction.append(pred)