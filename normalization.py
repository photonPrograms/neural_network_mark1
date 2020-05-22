import numpy as np

def normalize(X):
    """
    normalize the matrix of features X
    with mean and feature scaling by standard deviation
    """
    X = (X - np.mean(X)) / (np.std(X))
    return X
