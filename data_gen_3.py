# generate simple data with two features (x1, x2) and labels y
# for training neural network for classification
# the data can be separated by approximately a straight line

import random
import json
#import matplotlib.pyplot as plt

def two_class_linear(X):
    """
    create m datapoints with two class labels 0 and 1
    that are separable by approximately a straight line
    """
    # slope and y-intercept for the line
    slope1, c1 = -1, 0.3
    slope2, c2 = 1, -.5
    y = []
    for x in X:
        if (x[1] >= slope1 * x[0] + c1):
            y.append(2)
        elif (x[1] <= slope2 * x[0] + c2):
            y.append(1)
        else:
            y.append(0)
    return y

# minimum and maximum values on horizontal axis
min_x1, max_x1 = -2, 2
# mininimum and maximum values on vertical axis
min_x2, max_x2 = -2, 2

m = 250 # number of training examples

X = []
for i in range(m):
    X.append([min_x1 + (max_x1 - min_x1) * random.random(),
        min_x2 + (max_x2 - min_x2) * random.random()])
y = two_class_linear(X)
num_classes = 3 # number of classes

training_set = {
        "datapoint": X,
        "label": y,
        "num_classes": num_classes
}

filename = "training_set.json"
with open(filename, 'w') as f:
    json.dump(training_set, f)

"""
# plot the data points
X1, X2 = [], []
for x in X:
    X1.append(x[0])
    X2.append(x[1])

color = ["r", "g", "b", "k", "w"]
for i in range(m):
    plt.scatter(X1[i], X2[i], c = color[y[i]])
plt.show()
"""
