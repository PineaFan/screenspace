import sys
import time

import driver
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


def clamp(mi, ma, x):
    return max(mi, min(ma, x))


modes = {
    "01000": "draw",
    "00100": "close",
    "01100": "line",
    "01111": "rubber"
}

plt.ion()
plt.show()

driver = driver.Driver(debug=False, modules=["body"])
n = 0
while True:
    n += 1
    driver.calculate(1, 1)
    if driver.fullBodyResults is not None:
        bodyPoints = driver.fullBodyResults.landmark
        x = [p.x for p in bodyPoints]
        y = [p.y for p in bodyPoints]
        z = [p.z for p in bodyPoints]
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        # Show all the points
        for i in range(21):
            ax.scatter(x[i], y[i], z[i], color=("red"))
        plt.pause(0.001)

