import numpy as np

def sig(Z):
    """the sigmoid of a numpy array Z"""
    return 1 / (1 + np.exp(-Z))

def sigd(Z):
    """the derivative of sigmoid of a numpy array Z"""
    S = sig(Z)
    return S * (1 - S)
