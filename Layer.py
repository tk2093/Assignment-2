# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_pass(self, x):
        pass

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_pass(self, error, alpha=0.001):
        pass
