from Layer import Layer

import numpy as np


class Tanh(Layer):
    def __init__(self, x):
        self.input = x

    def forward_pass(self):
        return np.tanh(self.input)

    def backward_pass(self, error, alpha=0.001):
        out = (1 - np.tanh(self.input) ** 2)*error
        return out