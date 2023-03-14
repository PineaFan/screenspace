# import body
import sys
import cv2
import numpy as np
from driver import Driver
from hands import (Fist, HandModel, IndexFinger, MiddleFinger, Peace,
                   PinkyFinger, RingFinger, Spread, handToName)

width = 300
height = 150
scale = 2
width, height = width * scale, height * scale

driver = Driver(debug=False, modules=["hands"],
                flip_horizontal=True, flip_vertical=False)


class Colours:
    red: tuple = driver.hex_to_bgr("#FF0000")
    green: tuple = driver.hex_to_bgr("#00FF00")
    blue: tuple = driver.hex_to_bgr("#0000FF")
    yellow: tuple = driver.hex_to_bgr("#FFFF00")
    cyan: tuple = driver.hex_to_bgr("#00FFFF")
    magenta: tuple = driver.hex_to_bgr("#FF00FF")
    white: tuple = driver.hex_to_bgr("#FFFFFF")
    black: tuple = driver.hex_to_bgr("#020202")
    transparent: tuple = driver.hex_to_bgr("#000000")


actions = {
    IndexFinger().name: "draw",
    Peace().name: "line",
    Spread().name: "erase",
    MiddleFinger().name: "quit"
}


# Create a background
backgroundColour = Colours.white
background = np.zeros((height, width, 3), np.uint8)
background[:] = backgroundColour

# This will be an overlay over the background
currentDrawing = np.zeros((height, width, 3), np.uint8)
currentDrawing[:] = backgroundColour

# This is what the user is currently drawing, such as a line, and can be cleared
# If the user draws a line, it will be added here as a "preview"
currentMotion = np.zeros((height, width, 3), np.uint8)
currentMotion[:] = Colours.transparent

# What the user is currently doing, such as draw, line, erase, etc.
currentAction = []
currentPath = []

previousHands, knownHands = [], []

while True:
    currentFrame = background.copy()

    # Preprocessing
    # Overlay the current drawing. Use black areas as a mask
    for i in range(3):
        currentFrame[:, :, i] = np.where(
            currentDrawing[:, :, i] == 0, currentFrame[:, :, i], currentDrawing[:, :, i])

    # Calculate new matrices
    driver.calculate(background.shape[1], background.shape[0])

    # Render the stylus
    if driver.stylusCoords is not None:
        cv2.circle(
            currentDrawing if driver.stylusDraw else frame,
            (round(driver.stylusCoords[0]), round(driver.stylusCoords[1])), 3, (255, 0, 255), -1
        )
    # If there are hands on screen
    elif driver.screenspaceHandPoints:
        toRender = [4, 8, 12, 16, 20]  # Tips of each finger

        # If there are more last frame than this fame, remove the extra ones
        if len(previousHands) > len(driver.screenspaceHandPoints):
            previousHands = previousHands[:len(driver.screenspaceHandPoints)]
            knownHands = knownHands[:len(driver.screenspaceHandPoints)]
        # If there are more this frame than last frame, add the extra ones
        elif len(previousHands) < len(driver.screenspaceHandPoints):
            for i in range(len(previousHands), len(driver.screenspaceHandPoints)):
                previousHands.append([False, False, False, False, False, 0])
                knownHands.append(None)

        # Loop over each hand
        highestHandIndex = 0
        for handIndex, handData in enumerate(driver.screenspaceHandPoints):
            highestHandIndex = handIndex
            # Draw a circle on each point
            for pointIndex, point in enumerate(handData):
                if pointIndex in toRender:
                    isFingerRaised = driver.raisedFingers[handIndex][toRender.index(pointIndex)] > 0
                    cv2.circle(
                        currentFrame,
                        (round(point[0]), round(point[1])), 3, (Colours.green if isFingerRaised else Colours.red), -1
                    )
            # If fingers are different, reset the time they have been visible for
            fingerBooleans = [n > 0 for n in driver.raisedFingers[handIndex]]
            if previousHands[handIndex][:5] != fingerBooleans:
                previousHands[handIndex] = fingerBooleans + [0]
            else:
                # Add one to the time it has been visible
                previousHands[handIndex][5] += 1
            if previousHands[handIndex][-1] > 5:
                # If it has been visible for more than 5 frames, it is now known
                knownHands[handIndex] = HandModel(previousHands[handIndex][:5])
        # Remove extra hands
        knownHands = knownHands[:highestHandIndex + 1]

    currentAction = []
    for hand in knownHands:
        if hand is not None:
            currentAction.append(actions.get(handToName(hand.value[:5]), None))
        else:
            currentAction.append(None)
    print(currentAction)
    if "quit" in currentAction:
        break
    for handIndex, hand in enumerate(knownHands):
        if hand is None:
            continue
        if "draw" in currentAction and len(driver.screenspaceHandPoints) > knownHands.index(hand):
            cv2.circle(currentDrawing, (
                    round(driver.screenspaceHandPoints[knownHands.index(hand)][8][0]),
                    round(driver.screenspaceHandPoints[knownHands.index(hand)][8][1])
                ),
                5, Colours.black, -1
            )
            cv2.imshow("currentDrawing", currentDrawing)
        if "erase" in currentAction and len(driver.screenspaceHandPoints) > knownHands.index(hand):
            focusAbout = [0, 8, 20]
            focus = (
                (driver.screenspaceHandPoints[knownHands.index(hand)][focusAbout[0]][0] + driver.screenspaceHandPoints[knownHands.index(hand)][focusAbout[1]][0] + driver.screenspaceHandPoints[knownHands.index(hand)][focusAbout[2]][0]) / 3,
                (driver.screenspaceHandPoints[knownHands.index(hand)][focusAbout[0]][1] + driver.screenspaceHandPoints[knownHands.index(hand)][focusAbout[1]][1] + driver.screenspaceHandPoints[knownHands.index(hand)][focusAbout[2]][1]) / 3
            )
            focus = (round(focus[0]), round(focus[1]))
            cv2.circle(currentDrawing, focus, 20, Colours.white, -1)
            cv2.imshow("currentDrawing", currentDrawing)
            # Show an outline of the eraser on the currentMotion
            # To do this, draw a filled circle, then draw a transparent circle over the top
            cv2.circle(currentMotion, focus, 20, Colours.magenta, -1)
            cv2.circle(currentMotion, focus, 18, Colours.transparent, -1)

    mask = cv2.inRange(currentMotion, Colours.transparent, Colours.transparent)
    currentFrame = cv2.add(currentFrame, currentMotion, mask=mask)

    cv2.imshow("currentMotion", currentMotion)
    cv2.imshow("currentFrame", currentFrame)
    cv2.imshow("background", background)
    currentMotion[:] = Colours.transparent

    driver.render(currentFrame)
