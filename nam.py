from Linear_layer import LinearLayer as Ll
from tanh import Tanh
from sigmoid import Sigmoid
import numpy as np
import pickle
from network import Network
import matplotlib.pyplot as plt


def mse(pred, target):
    return np.mean(np.power(target-pred, 2))


def mse_grad(pred, target):
    return 2*(pred-target)/target.size


def cross_entropy_loss(pred, target):
    return -target * np.log(pred)


def cross_entropy_loss_grad(pred, target):
    return target - pred


x_train = np.array([[[0,0]],
                    [[0,1]],
                    [[1,0]],
                    [[1,1]]])
y_train = np.array([[0],
                    [1],
                    [1],
                    [0]])
# # Forward Pass
#
# # Initializing input hidden Layer
# layer1 = Ll(2,2)
# activation1= Tanh()
# # Initializing output layer
# outputlayer = Ll(2,1)
# activation2 = Tanh()
#
# for j in range(1000):
#     for i in range(len(y_train)):
#         # forward pass
#
#         # Input hidden Layer
#         a1 = layer1.forward_pass(x_train[i:i+1, :])
#         z1 = activation1.forward_pass(a1)
#         # output layer
#         a2 = outputlayer.forward_pass(z1)
#         z2 = activation2.forward_pass(a2)
#
#         # Backward pass
#
#         # Output layer
#         error = mse_grad(z2, y_train[i:i+1, :])
#         error = activation2.backward_pass(error)
#         error = outputlayer.backward_pass(error)
#         # Input hidden Layer
#         error = activation1.backward_pass(error)
#         #error = \
#         error = layer1.backward_pass(error)
#
# prediction = []
#
# for i in range(len(y_train)):
#     pred = x_train[i:i+1, :]
#     print(i, pred)
#     pred = layer1.forward_pass(pred)
#     print(i, pred)
#     pred = activation1.forward_pass(pred)
#     print(i, pred)
#     pred = outputlayer.forward_pass(pred)
#     print(i, pred)
#     pred = activation2.forward_pass(pred)
#     print(i, pred)
#     prediction.append(pred)
#
# wlayer= [layer1.weights, outputlayer.weights]
#
# with open('xor_weights.w', 'wb') as fp:
#     pickle.dump(wlayer, fp)
#
nn = Network()
layer1 = Ll(2,2)
nn.create(layer1)
act1 = Tanh()
nn.create(act1)
layer2 = Ll(2,1)
nn.create(layer2)
act2 = Tanh()
nn.create(act2)

nn.losses(mse, mse_grad)
loss = nn.fit(x_train, y_train, 1000, 4, 0.1)
pred = nn.predict(x_train)

plt.plot(loss)
plt.show()