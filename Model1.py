"""please run mnist_load.py before running this"""

from Linear_layer import LinearLayer as Ll
from Sequence import Sequence
from tanh import Tanh
from Softmax import Softmax
from Loss import MSE, XentLoss
import numpy as np
import pickle
import matplotlib.pyplot as plt


def accuracy(pred, target):
    acc = 0
    for i in range(len(target)):
        if np.argmax(pred[i]) == np.argmax(target[i]):
            acc +=1
    return acc/len(target)


# pickle saved by running mnist_load.py
with open("mnist.pkl", "br") as fh:
    data = pickle.load(fh)              # please run mnist_load.py before running this

X_train = data[0]
X_test = data[1]
X_cv = data[2]
y_train = data[3]
y_test = data[4]
y_cv = data[5]

nn = Sequence()
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

nn.losses(MSE)

train_loss, val_loss = nn.fit(X_train, y_train, 32, 0.1, X_cv, y_cv, stop_train=5, brk=5000) # model may run for a very long time, please put "brk = 1000" to break after 1000 steps
pred = nn.predict(X_cv)

plt.plot(train_loss)
plt.title('Train Loss for Model 1')
plt.savefig('Model1_Train_loss.png')
plt.show()
plt.plot(val_loss)
plt.title('Validation Loss for Model 1')
plt.savefig('Model1_val_loss.png')
plt.show()

val_acc = accuracy(np.array(pred),y_cv)
test_acc = accuracy(np.array(nn.predict(X_test)), y_test)


#saving weights
weights = []
for i in range(0,len(nn.layers),2):
    weights.append(nn.layers[i].weights)

with open('Model1_weights.w', 'wb') as file:
    pickle.dump(weights, file)

# Uncomment to load weights and make predictions

# loading weights
#
# with open('model1_weights.w', 'rb') as file:
#     load_weights = pickle.load(file)
# j=0
# for i in range(0,len(nn.layers),2):
#     nn.layers[i].weights = load_weights[j]
#     j += 1

# test_acc = accuracy(np.array(nn.predict(X_test)), y_test)

