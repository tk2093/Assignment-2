from Layer import Layer
import numpy as np


class XentLoss(Layer):

    def forward(pred, target):
        return np.mean(-target * np.log(pred))

    def backward(pred, target):
        return target - pred


class MSE(Layer):
    def forward(pred, target):
        #self.input = pred
        return np.mean(np.power(target-pred, 2))

    def backward(pred, target):
        #self.output = 2*(pred-target)/target.size
        return 2*(pred-target)/target.size
