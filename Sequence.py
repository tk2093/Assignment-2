import random
from Layer import Layer
import numpy as np


class Sequence(Layer):

    def __init__(self):
        self.layers = []
        self.error = None
        self.error_grad = None

    def losses(self, func):
        self.error = func
        self.error_grad = func

    def create(self, layer):
        self.layers.append(layer)

    def fit(self, X, y, batch_size=32, alpha=0.01, x_val=None, y_val=None, stop_train=5, brk=1000):
        train_loss = [100]
        val_loss = []
        i, j = 0, 0
        #for step in range(steps):
        while i < stop_train+1:
            if j > brk: break
            j += 1
            print(j)
            batch_error = 0
            idx = random.sample(range(X.shape[0]), batch_size)
            for row in zip(X[idx],y[idx]):
                x_temp, y_temp = row
                y_pred_temp = x_temp
                for layer in self.layers:
                    y_pred_temp = layer.forward_pass(y_pred_temp)

                batch_error += self.error.forward(y_pred_temp, y_temp.reshape(y_pred_temp.shape))

                dz = self.error_grad.backward(y_pred_temp, y_temp.reshape(y_pred_temp.shape))
                for layer in reversed(self.layers):
                    dz = layer.backward_pass(dz, alpha)
            if round((batch_error/batch_size),4) >= round(train_loss[-1],4):
                i += 1
            else:
                i = 0
            train_loss.append(batch_error/batch_size)
            if x_val is not None and ((j % 50) == 0):
                val_loss.append(self.error.forward(np.array(self.predict(x_val)).reshape(y_val.shape),y_val))
        if x_val is not None:
            return train_loss[1:], val_loss
        else:
            return train_loss[1:]

    def predict(self, test):
        pred = []
        for i in range(test.shape[0]):
            y = test[i]
            for layer in self.layers:
                y = layer.forward_pass(y)
            pred.append(y)

        return pred


