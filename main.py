from driver import Driver
import hands
import cv2
import numpy as np

driver = Driver(debug=False, modules=["hands"])
# Create a red image
background = np.zeros((150, 300, 3), np.uint8)
background[:] = driver.hex_to_bgr("#F27878")

debug = False

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
