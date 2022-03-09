from Layer import Layer
import numpy as np


class Softmax(Layer):
    def __init__(self, input_size):
        self.input_size = input_size

    def forward_pass(self, x):
        self.input = x
        tmp = np.exp(x)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward_pass(self, dz, learning_rate):
        input_error = np.zeros(dz.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(dz, np.identity(self.input_size) - out)