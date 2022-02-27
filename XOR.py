import numpy as np
import Layer
import Linear_layer as ll
import Softmax as Sftmx
import Cross_Entropy_Loss as XEloss


def cross_entropy_loss(pred, target):
    return -target * np.log(pred)


def cross_entropy_loss_grad(pred, target):
    return target - pred


x_train = np.array([[[0,0]],
                    [[0,1]],
                    [[1,0]],
                    [[1,1]]])
y_train = np.array([[[0]],
                    [[1]],
                    [[1]],
                    [[0]]])

#start_idx = batch_idx
#end_idx = start_idx + batch_size if start_idx + batch_size < samples.shape[0] else samples.shape[0]
#batch_samples = samples[start_idx: end_idx]
#batch_targets = get_one_hot(targets[start_idx: end_idx].astype(int), 10)
losses = []
correct = 0
samples = len(x_train)
for i in range(samples):

    # Forward Pass
    layer1_a = Layer.forward_pass(x_train[i])
    layer1_z = Sftmx.softmax(layer1_a)

    loss = XEloss.cross_entropy_loss(layer1_z.T, y_train[i])

    loss = np.sum(np.mean(loss, axis=0))
    losses.append(loss)

    correct += np.sum(np.argmax(layer1_z, axis=0) == y_train[i].astype(int))

    # Backward Pass
    d_loss = XEloss.cross_entropy_grad(layer1_z.T, y_train[i])
    # d_layer1_z = softmax_grad(layer1_z.T, d_loss) # Not necessary! `cross_entropy_grad` includes the softmax derivative.
    d_layer1_weights = d_loss.T @ np.concatenate((np.ones((batch_samples.shape[0], 1)), batch_samples), axis=1)

    # Gradient step
    layer1_weights = ll.backward_pass(layer1_z, y_train[i])