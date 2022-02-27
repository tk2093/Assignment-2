from Layer import Layer
import numpy as np


class Softmax(Layer):

    def forward_pass(self, x):
        self.input = x
        num = np.exp(self.input - np.max(self.input))
        self.output = num / np.sum(num, axis=0, keepdims=True)
        return self.output

    def backward_pass(self, probs, bp_err):
        self.output = probs
        dim = self.output.shape[1]
        output = np.empty(self.output.shape)
        for j in range(dim):
            d_prob_over_xj = - (self.output * self.output[:, [j]])  # i.e. prob_k * prob_j, no matter k==j or not
            d_prob_over_xj[:, j] += self.output[:, j]  # i.e. when k==j, +prob_j
            output[:, j] = np.sum(bp_err * d_prob_over_xj, axis=1)
        return output
