import json
import matplotlib.pyplot as plt

filename = "costdata.json"
with open(filename) as f:
    J = json.load(f)

plt.plot(J)
plt.show()
