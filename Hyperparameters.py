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

nn = Network()
layer1 = Ll(784,400)
layer1.weights = np.zeros((785,400))
nn.create(layer1)
act1 = Tanh()
nn.create(act1)
layer2 = Ll(400,100)
layer2.weights = np.zeros((401,100))
nn.create(layer2)
act2 = Tanh()
nn.create(act2)
layer3 = Ll(100,50)
layer3.weights = np.zeros((101,50))
nn.create(layer3)
act3 = Tanh()
nn.create(act3)
layer4 = Ll(50,10)
layer4.weights = np.zeros((51,10))
nn.create(layer4)
act4 = Softmax(10)
nn.create(act4)


nn.losses(mse, mse_grad)
train_loss, val_loss = nn.fit(X_train, y_train, 32, 0.01, X_cv, y_cv, brk=5000) # model may run for a very long time, please put "brk = 1000" to break after 1000 steps
pred = nn.predict(X_cv)
val_error = mse(np.array(pred).reshape(np.array(pred).shape[0],10),y_cv)
plt.plot(train_loss)
plt.title('Train Loss for Model')
plt.savefig('Model_Train_loss.png')
plt.show()
plt.plot(val_loss)
plt.title('Validation Loss for Model')
plt.savefig('Model_val_loss.png')
plt.show()

val_acc = accuracy(np.array(pred),y_cv)
test_acc = accuracy(np.array(nn.predict(X_test)), y_test)


#saving weights
weights = []
for i in range(0,len(nn.layers),2):
    weights.append(nn.layers[i].weights)

with open('Model_weights.w', 'wb') as file:
    pickle.dump(weights, file)
