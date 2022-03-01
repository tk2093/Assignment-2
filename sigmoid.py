from Layer import Layer
import numpy as np


class Sigmoid(Layer):
#    def __init__(self):
#        self.input = x

    def forward_pass(self, x):
        self.input = x
        self.output = 1. / (1. + np.exp(-self.input))
        return self.output

    def backward_pass(self, error, alpha=0.001):
        return self.output * (1 - self.output) * error
