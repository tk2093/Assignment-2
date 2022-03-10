from Linear_layer import LinearLayer as Ll
from tanh import Tanh
from sigmoid import Sigmoid
from Loss import MSE, XentLoss
import numpy as np
import pickle
from Sequence import Sequence
import matplotlib.pyplot as plt


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

x_train = x_train.reshape(x_train.shape[0],1,2)
nn = Sequence()
layer1 = Ll(2,2)
nn.create(layer1)
act1 = Tanh()
nn.create(act1)
layer2 = Ll(2,1)
nn.create(layer2)
act2 = Tanh()
nn.create(act2)

nn.losses(MSE)
loss = nn.fit(x_train, y_train, 4, 0.1, stop_train = 5)  #Please run again if stuck at step<300
pred = nn.predict(x_train)
error = MSE.forward(np.array(pred).reshape(4,1),y_train)
plt.plot(loss)
plt.show()
print('predicitons:',np.array(pred).reshape(4,1))
#saving weights
weights = []
for i in range(0,len(nn.layers),2):
    weights.append(nn.layers[i].weights)

with open('xor_weights.w', 'wb') as file:
    pickle.dump(weights, file)

#Uncomment to load weights and make predictions
#loading weights
#
# with open('xor_weights.w', 'rb') as file:
#     load_weights = pickle.load(file)
# j=0
# for i in range(0,len(nn.layers),2):
#     nn.layers[i].weights = load_weights[j]
#     j += 1

# pred = nn.predict(x_train.reshape(x_train.shape[0],1,2))
# error = MSE.forward(np.array(pred).reshape(4,1),y_train)