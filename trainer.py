import json
import numpy as np

from normalization import normalize
from sigmoid import sig, sigd
from cost import cost_func

filename = "training_set.json"
with open(filename) as f:
    training_set = json.load(f)

X = np.array(training_set["datapoint"])

# number of training examples
m = np.size(X, axis = 0)

# number of classes in output
k = training_set["num_classes"]

# number of input features excluding bias node
n = np.size(X, axis = 1)

# X is a matrix with m rows
# each row containing the features including the bias node
normalize(X)
X = np.hstack((np.ones((m, 1)), X))

# Y is a matrix with m rows
# each row with 0 or 1 for each output class
Y = np.zeros((m, k))
for i in range(m):
    Y[i][training_set["label"][i]] = 1

# parameters for multiplying with first and second layers
h = n + 2 # hidden layer size
Th1 = np.random.rand(n + 1, h - 1)
Th2 = np.random.rand(h, k)

lamb = 0.01 # regularization parameter
niter = 100000 # number of iterations for learning
alpha = 0.01 # learning rate

# cost function
J = []

for N in range(niter):
    # forward propagation
    Z2 = X @ Th1
    A2 = np.hstack((np.ones((m, 1)), sig(Z2)))
    Z3 = A2 @ Th2
    A3 = sig(Z3)

    J.append(cost_func(Y, A3, Th1, Th2, m, lamb))

    # back propagation
    del3 = A3 - Y
    del2 = ((del3 @ np.transpose(Th2)) * np.hstack((np.ones((m, 1)), sigd(Z2))))[:, 1:]

    Th2_grad = 1 / m * (np.transpose(A2) @ del3)
    Th1_grad = 1 / m * (np.transpose(X) @ del2)

    Th2_grad += lamb / m * np.hstack((np.zeros((h, 1)), Th2[:, 1:]))
    Th1_grad += lamb / m * np.hstack((np.zeros((n + 1, 1)), Th1[:, 1:]))

    Th2 -= alpha * Th2_grad
    Th1 -= alpha * Th1_grad

filename = "costdata.json"
with open(filename, 'w') as f:
    json.dump(J, f)

params = {
    "1": Th1.tolist(),
    "2": Th2.tolist()
}
filename = "parameters.json"
with open(filename, 'w') as f:
    json.dump(params, f)
