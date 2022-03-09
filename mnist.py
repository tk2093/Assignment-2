from Linear_layer import LinearLayer as Ll
from network import Network
from tanh import Tanh
from sigmoid import Sigmoid
import numpy as np
import pickle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
#from keras.datasets import mnist
import matplotlib.pyplot as plt



def get_one_hot(targets_in, nb_classes):
    res = np.eye(nb_classes)[np.array(targets_in).reshape(-1)]
    return res.reshape(list(targets_in.shape)+[nb_classes])

def mse(pred, target):
    return np.mean(np.power(target-pred, 2))


def mse_grad(pred, target):
    return 2*(pred-target)/target.size


def cross_entropy_loss(pred, target):
    return -target * np.log(pred)


def cross_entropy_loss_grad(pred, target):
    return target - pred


samples, targets = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
targets = targets.astype(int)
X_train, X_test, y_train, y_test = train_test_split(samples, targets, test_size=10000, random_state=42)
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train/255.
X_test = X_test/255.

y_train = get_one_hot(y_train, 10)
y_test = get_one_hot(y_test, 10)

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# # Initializing input  Layer
# layer1 = Ll(784,100)
# activation1= Tanh()
# # intermediate hidden layer
# layer2 = Ll(100,50)
# activation2 = Tanh()
# #output layer
# layer3 = Ll(50,10)
# activation3 = Tanh()
#
# batch_size = 32
# correct = 0
# lr = 0.01
# losses = []
# for batch_idx in range(0, X_train.shape[0], batch_size):
#     start_idx = batch_idx
#     end_idx = start_idx + batch_size if start_idx + batch_size < samples.shape[0] else samples.shape[0]
#     batch_samples = X_train[start_idx: end_idx]
#     batch_targets = y_train[start_idx: end_idx]
#
#     # Forward Pass
#     #layer1
#     a1 = layer1.forward_pass(batch_samples)
#     z1 = activation1.forward_pass(a1)
#     #layer2
#     a2 = layer2.forward_pass(z1)
#     z2 = activation2.forward_pass(a2)
#     #layer3
#     a3 = layer3.forward_pass(z2)
#     z3 = activation3.forward_pass(a3)
#
#     error = mse_grad(z3, batch_targets)
#
#     #loss = cross_entropy_loss(z3.T, batch_targets)
#     #loss = np.sum(np.mean(loss, axis=0))
#     #losses.append(loss)
#     #correct += np.sum(np.argmax(z3, axis=0) == y_train[start_idx: end_idx].astype(int))
#
#     #backward pass
#     #layer3
#     error = activation3.backward_pass(error)
#     error = layer3.backward_pass(error)
#     #Layer2
#     error = activation2.backward_pass(error)
#     error = layer2.backward_pass(error)
#     #Layer1
#     error = activation1.backward_pass(error)
#     error = layer1.backward_pass(error)
#
# pred = []
# #for i in X_cv:
# a1 = layer1.forward_pass(X_cv)
# z1 = activation1.forward_pass(a1)
# #layer2
# a2 = layer2.forward_pass(z1)
# z2 = activation2.forward_pass(a2)
# #layer3
# a3 = layer3.forward_pass(z2)
# z3 = activation3.forward_pass(a3)
#     #pred.append(z3)
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
loss = nn.fit(X_train, y_train, 1000, 32, 0.1)
pred = nn.predict(X_cv)



plt.plot(loss)
plt.show()