import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def get_one_hot(targets_in, nb_classes):
    res = np.eye(nb_classes)[np.array(targets_in).reshape(-1)]
    return res.reshape(list(targets_in.shape)+[nb_classes])


samples, targets = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
samples = samples.reshape(samples.shape[0],1,784)
targets = targets.astype(int)
X_train, X_test, y_train, y_test = train_test_split(samples, targets, test_size=10000, random_state=42)


X_train = X_train/255.
X_test = X_test/255.

y_train = get_one_hot(y_train, 10)
y_test = get_one_hot(y_test, 10)

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

with open("mnist.pkl", "bw") as fh:
    data = (X_train,
            X_test,
            X_cv,
            y_train,
            y_test,
            y_cv)
    pickle.dump(data, fh)