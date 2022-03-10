from Layer import Layer
import numpy as np


class Softmax(Layer):
    def __init__(self, size):
        self.size = size

    def forward_pass(self, x):
        self.input = x
        num = np.exp(x)
        self.output = num / np.sum(num)
        return self.output

    def backward_pass(self, dz, learning_rate):
        out = np.tile(self.output.T, self.size)
        return self.output * np.dot(dz, np.identity(self.size) - out)