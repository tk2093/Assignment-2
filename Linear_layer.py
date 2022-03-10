from Layer import Layer
import numpy as np


class LinearLayer(Layer):

    def __init__(self, n_neurons_in, n_neurons_out):
        self.weights = np.random.rand(n_neurons_in+1, n_neurons_out) - 0.5

    def forward_pass(self, x):
        self.input = x
        a = np.concatenate((np.ones((self.input.shape[0], 1)), self.input), axis=1) @ self.weights
        return a

    def backward_pass(self, dz, alpha=0.1):
        d_layer_weights = np.concatenate((np.ones((self.input.shape[0], 1)), self.input), axis=1).T @ dz
        self.weights = self.weights - alpha * d_layer_weights
        #print(self.weights)
        self.output = np.dot(dz, self.weights[1:,:].T)
        return self.output
