# import body
import cv2
import hands
import numpy as np
from driver import Driver
from hands import (Fist, HandModel, IndexFinger, MiddleFinger, Peace,
                   PinkyFinger, RingFinger, Spread)

width = 300
height = 150
scale = 2
width, height = width * scale, height * scale

driver = Driver(debug=False, modules=["hands"], flip_horizontal=False, flip_vertical=False)
# Create a default background
background = np.zeros((height, width, 3), np.uint8)  # 300 wide, 150 high, 3 channels (RGB).
# Translucency is not supported due to masking issues
# Black is for transparent
background[:] = driver.hex_to_bgr("#FFFFFF")
# background[:] = driver.hex_to_bgr("#F27878")
# background[:] = driver.hex_to_bgr("#000000")

currentDrawing = np.zeros((height, width, 3), np.uint8)
currentDrawing[:] = [255, 255, 255]

debug = False

class Colours:
    on: tuple = (0, 255, 0)
    off: tuple = (0, 0, 255)


previousHands = []
knownHands = []

x = 0
while True:
    frame = background.copy()
    # Overlay the current drawing. Use white areas as a mask
    mask = cv2.inRange(currentDrawing, (254, 254, 254), (255, 255, 255))
    # Then overlay everything else. If the mask is white, the pixel is not drawn, otherwise it is in the correct color
    for i in range(3):
        frame[:, :, i] = np.where(mask == 255, frame[:, :, i], currentDrawing[:, :, i])

    driver.calculate(background.shape[1], background.shape[0])
    if driver.stylusCoords is not None:
        cv2.circle(currentDrawing if driver.stylusDraw else frame, (round(driver.stylusCoords[0]), round(driver.stylusCoords[1])), 3, (255, 0, 255), -1)
    elif driver.screenspaceHandPoints:  # When hands are detected
        toRender = [4, 8, 12, 16, 20]
        # If there are more hands in the list of previously stored ones than there are hands detected, remove the extra ones
        if len(previousHands) > len(driver.screenspaceHandPoints):
            previousHands = previousHands[:len(driver.screenspaceHandPoints)]
            knownHands = knownHands[:len(driver.screenspaceHandPoints)]
        # Loop over each hand
        maxHand = 0
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
            maxHand = handID
        knownHands = knownHands[:maxHand + 1]
    # if driver.fullBodyCoordinates:
    #     frame = body.renderBody(frame, driver.bodyCoordinates)
    if any([n == MiddleFinger() for n in knownHands]):
        print("\033[91mMiddle finger detected, exiting")
        import sys
        sys.exit()
    for hand in knownHands:
        if hand == IndexFinger():
            if len(driver.screenspaceHandPoints) > knownHands.index(hand):
                cv2.circle(currentDrawing, (
                        round(driver.screenspaceHandPoints[knownHands.index(hand)][8][0]),
                        round(driver.screenspaceHandPoints[knownHands.index(hand)][8][1])
                    ),
                    5,
                    driver.hex_to_bgr("#020202"), -1
                )
                cv2.imshow("currentDrawing", currentDrawing)
    if debug:
        frame = cv2.resize(frame, (1500, 1000))
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    else:
        driver.render(frame)
