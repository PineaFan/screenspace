from driver import Driver
import cv2
import time
import manipulation
import numpy as np

driver = Driver(debug=True, modules=["hands"])
background = np.zeros([1, 1, 1], dtype=np.uint8)
background.fill(255)
background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
background = cv2.resize(background, (150, 100))

x = 0
while True:
    frame = background.copy()
    driver.calculate(background.shape[1], background.shape[0])
    # Show point 8 from the screenspace hand points
    if driver.screenspaceHandPoints:
        point = driver.screenspaceHandPoints[8]
        cv2.rectangle(frame, (5, round(point[1] - 10)), (10, round(point[1] + 10)), (1, 1, 1, 1), -1)

    driver.render(frame)
