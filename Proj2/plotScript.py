import json
import pylab as plt

accuracies = open("accuracies", "r")
epochs = 100

colors = ["green", "blue", "orange"]

plt.figure("accuracies")
i = 0
for line in accuracies:
    type, list = line.split(":")
    list = json.loads(list)
    plt.plot(range(epochs), list, color = colors[i], label = type)
    i += 1

plt.legend(loc = "best")
plt.show()

