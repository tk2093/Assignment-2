from Linear_layer import LinearLayer as Ll
from network import Network
from tanh import Tanh
from Softmax import Softmax
from sigmoid import Sigmoid
import numpy as np
import pickle
import matplotlib.pyplot as plt


def mse(pred, target):
    return np.mean(np.power(target-pred, 2))


def mse_grad(pred, target):
    return 2*(pred-target)/target.size


def cross_entropy_loss(pred, target):
    return -target * np.log(pred)


def cross_entropy_loss_grad(pred, target):
    return target - pred


def accuracy(pred, target):
    acc = 0
    for i in range(len(target)):
        if np.argmax(pred[i]) == np.argmax(target[i]):
            acc +=1
    return acc/len(target)


# pickle saved by running mnist_load.py
with open("mnist.pkl", "br") as fh:
    data = pickle.load(fh)

X_train = data[0]
X_test = data[1]
X_cv = data[2]
y_train = data[3]
y_test = data[4]
y_cv = data[5]

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

# Model 1: Hidden layer 1 total layers 3, learning rate 0.1, Batch size 32, activation: Tanh and Softmax
nn = Network()
layer1 = Ll(784,400)
nn.create(layer1)
act1 = Tanh()
nn.create(act1)
layer2 = Ll(400,100)
nn.create(layer2)
act2 = Tanh()
nn.create(act2)
layer3 = Ll(100,50)
nn.create(layer3)
act3 = Tanh()
nn.create(act3)
layer4 = Ll(50,10)
nn.create(layer4)
act4 = Softmax(10)
nn.create(act4)

nn.losses(mse, mse_grad)
train_loss, val_loss = nn.fit(X_train, y_train, 32, 0.1, X_cv, y_cv)
pred = nn.predict(X_cv)
val_error = mse(np.array(pred).reshape(np.array(pred).shape[0],10),y_cv)
plt.plot(train_loss)
plt.title('Train Loss')
plt.savefig('Train_loss.png')
plt.show()
plt.plot(val_loss)
plt.title('Validation Loss')
plt.savefig('val_loss.png')
plt.show()

val_acc = accuracy(np.array(pred),y_cv)
test_acc = accuracy(np.array(nn.predict(X_test)), y_test)


#saving weights
weights = []
for i in range(0,len(nn.layers),2):
    weights.append(nn.layers[i].weights)

with open('mnist_weights.w', 'wb') as file:
    pickle.dump(weights, file)

#Uncomment to load weights and make predictions
#loading weights
#
# with open('mnist_weights.w', 'rb') as file:
#     load_weights = pickle.load(file)
# j=0
# for i in range(0,len(nn.layers),2):
#     nn.layers[i].weights = load_weights[j]
#     j += 1

# pred = nn.predict(x_train.reshape(x_train.shape[0],1,2))
# error = mse(np.array(pred).reshape(4,1),y_train)

