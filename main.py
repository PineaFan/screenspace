from driver import Driver
import cv2
import matplotlib.pyplot as plt
import numpy as np

driver = Driver(debug=True, modules=["hands"])
# Create a red image
background = np.zeros((150, 300, 3), np.uint8)
# background[:] = (255, 0, 255)  # Magenta
# background[:] = (0, 0, 0)  # Black (transparent)
background[:] = (120, 120, 242)  # Red

debug = False

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x = 0
while True:
    frame = background.copy()
    driver.calculate(background.shape[1], background.shape[0])

    if driver.screenspaceHandPoints:
        toRender = [4, 8, 12, 16, 20]
        for x in driver.screenspaceHandPoints:
            for i in [n for n in toRender if n < len(x)]:
                point = x[i]
                cv2.circle(frame, (round(point[0]), round(point[1])), 5, (0, 255, 0), -1)

    if debug:
        frame = cv2.resize(frame, (1500, 1000))
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    else:
        driver.render(frame)
