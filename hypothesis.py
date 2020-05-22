import numpy as np

from sigmoid import sig

def hypothesis(x, Th1, Th2):
    """
    calculate the output layer of the neural network
    and return the most probable class for the datapoint x
    x does not contain the bias node
    using parameter matrices Th1 and Th2
    arguments are lists not numpy arrays
    """
    x.insert(0, 1)
    x, Th1, Th2 = np.array(x), np.array(Th1), np.array(Th2)
    a2 = np.hstack(([1], sig(x @ Th1)))
    a3 = sig(a2 @ Th2)
    return np.argmax(a3)
