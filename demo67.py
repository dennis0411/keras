import numpy as np
from tensorflow import nn

scores = [3.0, 1.0, 2.0]


def normalRatio(x):
    x = np.array(x)
    return x / np.sum(x)


def mySoftMax(x):
    x = np.array(x)
    return np.exp(x) / np.sum(np.exp(x))


print(normalRatio(scores))
print(mySoftMax(scores))
print(nn.softmax(scores).numpy())


