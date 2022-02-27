from Layer import Layer
import numpy as np


class LinearLayer(Layer):

    def __init__(self, n_neurons_in, n_neurons_out):
        self.weights = np.random.rand(n_neurons_out, n_neurons_in+1) - 0.5

    # returns output for a given input
    def forward_pass(self, x):
        self.input = x
        a = self.weights @ np.concatenate((np.ones((self.input.shape[0], 1)), self.input), axis=1).T
        return a

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_pass(self, error, alpha=0.001):

        d_layer_weights = error.T @ np.concatenate((np.ones((self.input.shape[0], 1)), self.input), axis=1)
        # update parameters
        self.weights = self.weights + alpha * d_layer_weights
        error_in = np.dot(error, self.weights[:,1:])
        return error_in
