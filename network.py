import random


class Network:

    def __init__(self):
        self.layers = []
        self.error = None
        self.error_grad = None

    def losses(self, func, func_grad):
        self.error = func
        self.error_grad = func_grad

    def create(self, layer):
        self.layers.append(layer)

    def fit(self, X, y, steps, batch_size, alpha=0.01):
        loss = []
        for step in range(steps):
            batch_error = 0
            idx = random.sample(range(X.shape[0]), batch_size)
            for row in zip(X[idx],y[idx]):
                x_temp, y_temp = row
                y_pred_temp = x_temp
                for layer in self.layers:
                    y_pred_temp = layer.forward_pass(y_pred_temp)

                batch_error += self.error(y_pred_temp, y_temp)

                dz = self.error_grad(y_pred_temp, y_temp)
                for layer in reversed(self.layers):
                    dz = layer.backward_pass(dz, alpha)
            loss.append(batch_error/batch_size)
        return loss

    def predict(self, test):
        pred = []
        for i in range(test.shape[0]):
            y = test[i]
            for layer in self.layers:
                y = layer.forward_pass(y)
            pred.append(y)

        return pred


