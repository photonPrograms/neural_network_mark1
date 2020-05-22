import json
import matplotlib.pyplot as plt
import numpy as np
import random
from hypothesis import hypothesis

# reading training dataset
filename = "training_set.json"
with open(filename) as f:
    tset = json.load(f)

# reading the parameters
filename = "parameters.json"
with open(filename) as f:
    params = json.load(f)

plt.figure()
colors = ['r', 'g', 'b', 'c', 'y', 'm', 'w', 'k']

# plotting the training_set
plt.subplot(121)
for i in range(len(tset["datapoint"])):
    plt.scatter(tset["datapoint"][i][0], tset["datapoint"][i][1], color = colors[tset["label"][i]])

# plotting random points for testing
plt.subplot(122)

X = np.array(tset["datapoint"])

# maximum, minimum, midpoint, halfrange of training set
minx1, maxx1 = np.min(X[:, 0]), np.max(X[:, 0])
midx1, hrx1 = (minx1 + maxx1) / 2, (maxx1 - minx1) / 2
minx2, maxx2 = np.min(X[:, 1]), np.max(X[:, 1])
midx2, hrx2 = (minx2 + maxx2) / 2, (maxx2 - minx2) / 2

# extending the range of neural network beyond test data
ext = 1.25

for i in range(200):
    x1 = random.random() * ext * hrx1 * random.choice([-1, 1]) + midx1
    x2 = random.random() * ext * hrx2 * random.choice([-1, 1]) + midx2
    plt.scatter(x1, x2, color = colors[hypothesis([x1, x2], params["1"], params["2"])])

plt.show()
