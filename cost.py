import numpy as np

def cost_func(Y, A3, Th1, Th2, m, lamb):
    unreg_cost = -1 / m * np.sum(Y * np.log(A3) + (1 - Y) * np.log(1 - A3))
    reg_cost = 1 / (2 * m) * (np.sum(Th1[:, 1:] ** 2) + np.sum(Th2[:, 1:] ** 2))
    return unreg_cost + reg_cost
