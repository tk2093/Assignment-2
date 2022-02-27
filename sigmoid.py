from Layer import Layer
import numpy as np


class Sigmoid(Layer):
    def __init__(self, x):
        self.input = x

    def forward_pass(self, x):
        self.input = x
        self.output = 1. / (1. + np.exp(-self.input))
        return self.output

    def backward_pass(self, out, grad_in):
        return self.output * (1 - self.output) * grad_in
