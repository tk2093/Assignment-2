from Layer import Layer
import numpy as np


class Tanh(Layer):
#    def __init__(self, x):
#        self.input = x

    def forward_pass(self, x):
        self.input = x
        return np.tanh(self.input)

    def backward_pass(self, dz, alpha=0.1):
        self.output = (1 - np.tanh(self.input) ** 2) * dz
        return self.output
