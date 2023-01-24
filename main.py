from driver import Driver
import cv2
import numpy as np

driver = Driver(debug=True, modules=["hands"])
# background = np.zeros([1, 1, 1], dtype=np.uint8)
# background.fill((255, 0, 0))
# background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
# Create a red image
background = np.zeros((150, 300, 3), np.uint8)
background[:] = (0, 0, 0)

debug = False

x = 0
while True:
    frame = background.copy()
    driver.calculate(background.shape[1], background.shape[0])

    if debug:
        frame = cv2.resize(frame, (1500, 1000))
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    else:
        driver.render(frame)
