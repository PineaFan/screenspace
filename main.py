from driver import Driver
import hands
import body
from hands import HandModel, Fist, Spread, Peace, IndexFinger, MiddleFinger, RingFinger, PinkyFinger
import cv2
import numpy as np

driver = Driver(debug=False, modules=["hands"])
# Create a red image
background = np.zeros((150, 300, 3), np.uint8)
background[:] = driver.hex_to_bgr("#F27878")
# background[:] = driver.hex_to_bgr("#000000")

debug = False

class Colours:
    on: tuple = (0, 255, 0)
    off: tuple = (0, 0, 255)


previousHands = []
knownHands = []

x = 0
while True:
    frame = background.copy()
    driver.calculate(background.shape[1], background.shape[0])

    if driver.screenspaceHandPoints:  # When hands are detected
        toRender = [4, 8, 12, 16, 20]
        # If there are more hands in the list of previously stored ones than there are hands detected, remove the extra ones
        if len(previousHands) > len(driver.screenspaceHandPoints):
            previousHands = previousHands[:len(driver.screenspaceHandPoints)]
            knownHands = knownHands[:len(driver.screenspaceHandPoints)]
        # Loop over each hand
        for handID, hand in enumerate(driver.screenspaceHandPoints):
            # If there are less hands than there are in the list, add a default one to the list
            if len(previousHands) <= handID:
                previousHands.append([False, False, False, False, False])
                knownHands.append(Fist())
            # Draw each point on the video
            for index, i in enumerate([n for n in toRender if n < len(hand)]):
                point = hand[i]
                cv2.circle(frame, (round(point[0]), round(point[1])), 3, Colours.on if driver.raisedFingers[handID][index] > 0 else Colours.off, -1)
            # Create a list of booleans by checking if the dot product of the finger and the palm is greater than 0
            booleans = [n > 0 for n in driver.raisedFingers[handID]]
            if previousHands[handID][:5] != booleans:  # If the fingers are different, reset the time visible
                previousHands[handID] = booleans + [0]
            else:  # Otherwise add to the time visible
                previousHands[handID][-1] += 1
            # If the hand has been visible for 5 frames, add it to the list of confirmed hands
            if previousHands[handID][-1] > 5:
                knownHands[handID] = HandModel(previousHands[handID])
    # if driver.fullBodyCoordinates:
    #     frame = body.renderBody(frame, driver.bodyCoordinates)
    if any([n == MiddleFinger() for n in knownHands]):
        print("\033[91mMiddle finger detected, exiting")
        import sys
        sys.exit()
    if debug:
        frame = cv2.resize(frame, (1500, 1000))
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    else:
        driver.render(frame)
