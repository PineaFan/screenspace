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
                flip_horizontal=False, flip_vertical=False)


class Colours:
    red: tuple = driver.hex_to_bgr("#F27878")
    green: tuple = driver.hex_to_bgr("#A1CC65")
    blue: tuple = driver.hex_to_bgr("#6576CC")
    yellow: tuple = driver.hex_to_bgr("#E6DC71")
    cyan: tuple = driver.hex_to_bgr("#71AEF5")
    magenta: tuple = driver.hex_to_bgr("#A358B3")
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

statuses = {
    "Calibration": [Colours.red, "No codes found, try moving around", 20],
    "Correcting": [Colours.yellow, "Some codes hidden, attempting to correct", 20],
    "Accurate": [Colours.green, "", 5]
}

currentOverlay = None
lastVisibility = False

while True:
    currentFrame = background.copy()
    if currentOverlay is None and driver.cameraFrame is not None:
        currentOverlay = np.zeros((driver.cameraFrame.shape[0], driver.cameraFrame.shape[1], 3), np.uint8)

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
        knownHands = knownHands[:highestHandIndex]

    currentAction = []
    for hand in knownHands:
        if hand is not None:
            currentAction.append(actions.get(handToName(hand.value[:5]), None))
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
            # Avoid list index out of range errors
            for i in range(len(focusAbout)):
                if focusAbout[i] >= len(driver.screenspaceHandPoints[knownHands.index(hand)]):
                    focusAbout[i] = len(driver.screenspaceHandPoints[knownHands.index(hand)]) - 1

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

    # Create a mask from the currentMotion, where all non black pixels become white
    mask = cv2.inRange(currentMotion, Colours.transparent, Colours.transparent)
    # Apply the mask to the currentFrame, so white pixels on the mask make the currentFrame black
    currentFrame = cv2.bitwise_and(currentFrame, currentFrame, mask=mask)

    for i in range(3):
        currentFrame[:, :, i] = np.where(
            currentMotion[:, :, i] == 0, currentFrame[:, :, i], currentMotion[:, :, i])

    currentMotion[:] = Colours.transparent

    # Only update when the status has been the same for 10 frames
    if ((driver.visibilityTime > 3 and driver.visibility != lastVisibility) or driver.visibilityTime > 10) and currentOverlay is not None:
        lastVisibility = driver.visibility
        statusName = driver.visibility
        status = statuses[statusName]
        if statusName == "Accurate":
            currentOverlay[:, :, :] = Colours.transparent
        cv2.rectangle(currentOverlay, (0, 0), (currentOverlay.shape[1], status[2]), status[0], -1)
        # Add text to the overlay, 20px high and white
        cv2.putText(currentOverlay, status[1], (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colours.white, 1, cv2.LINE_AA)
        cv2.imshow("currentOverlay", currentOverlay)

    driver.render(currentFrame, currentOverlay)
