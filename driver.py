# import body
import cv2
import hands
import manipulation
import numpy as np

import screenspace


class Driver:
    def __init__(self, debug=False, modules=[], flip_horizontal: bool = False, flip_vertical: bool = False):
        self.modules = modules
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

        self.cameraFrame = None
        self.currentFrame = None
        self.debug = debug
        self.warpMatrix = None
        self.inverseMatrix = None

        self.videospaceStylusCoords = []
        self.stylusCoords = (0, 0)
        self.stylusDraw = None

        self.handVideoCoordinates = None
        self.handNormalisedCoordinates = None
        self.screenspaceHandPoints = None
        self.fullHandResults = None
        self.fullHandLandmarks = None
        self.raisedFingers = []

        self.fullBodyResults = None
        self.bodyVideoCoordinates = None
        self.screenspaceBodyPoints = None

        self.screenspaceCorners = None
        self.screenspaceMidpoints = None
        self.screenspaceCenter = None

        self.previousFullCodes = [x for x in screenspace.defaultFullCodes]

        self.visibility = "Calibration"  # Calibration, Correcting, Accurate
        self.visibilityTime = 1000

    def hex_to_bgr(self, hex):
        hex = hex.lstrip('#')
        hlen = len(hex)
        return tuple(reversed([int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3)]))

    def calculate(self, width, height):
        frame = screenspace.getCurrentFrame()
        self.cameraFrame = frame.copy()
        dimensions = manipulation.createImageWithDimensions(width, height)
        self.screenspaceCorners, outputFrame, self.previousFullCodes, self.videospaceStylusCoords, self.stylusDraw, visibility = screenspace.getScreenspacePoints(frame, frame, self.debug, self.previousFullCodes)
        if visibility != self.visibility:
            self.visibility = visibility
            self.visibilityTime = 0
        else:
            self.visibilityTime += 1
        self.screenspaceMidpoints, outputFrame = screenspace.getMidpoints(self.screenspaceCorners, frame, self.debug)
        self.warpMatrix = manipulation.generateWarpMatrix(dimensions, self.screenspaceCorners)

        try:
            self.inverseMatrix = np.linalg.inv(self.warpMatrix)
        except np.linalg.LinAlgError:
            self.inverseMatrix = None

        centerX, centerY = frame.shape[1] // 2, frame.shape[0] // 2
        centerWarpMatrix = manipulation.generateWarpMatrix(frame, self.screenspaceCorners)
        self.screenspaceCenter = manipulation.findNewCoordinate((centerX, centerY), centerWarpMatrix)
        if len(self.videospaceStylusCoords):
            newCoords = [manipulation.findNewCoordinate(c, self.inverseMatrix) for c in self.videospaceStylusCoords]
            # Find the midpoint
            self.stylusCoords = round((newCoords[0][0] + newCoords[1][0]) / 2), round((newCoords[0][1] + newCoords[1][1]) / 2)
        else:
            self.stylusCoords = None

        outputFrame = screenspace.addScreenspaceOverlay(outputFrame, self.screenspaceCorners, self.debug)

        if "hands" in self.modules:
            handPoints, self.fullHandResults = hands.getHandPoints(frame)
            self.fullHandLandmarks = handPoints
            outputFrame = hands.renderHandPoints(outputFrame, self.fullHandResults, self.debug)
            screenspaceHandCoords = []
            self.screenspaceHandPoints = []
            if handPoints:
                self.raisedFingers = [hands.getExtendedFingers(hp) for hp in handPoints]
                self.handVideoCoordinates = [hands.toVideospaceCoords(h.landmark, outputFrame.shape[1], outputFrame.shape[0]) for h in handPoints]
                # To work out positions on screen, multiply by the warp matrix
                self.handNormalisedCoordinates = []
                for hand in self.handVideoCoordinates:
                    self.handNormalisedCoordinates.append([])
                    self.screenspaceHandPoints.append([])
                    for point in hand:
                        self.handNormalisedCoordinates[-1].append(manipulation.findNewCoordinate(point, self.warpMatrix))
                        if self.inverseMatrix is not None:
                            self.screenspaceHandPoints[-1].append(manipulation.findNewCoordinate(point, self.inverseMatrix))
            outputFrame = hands.renderHandPoints(outputFrame, self.fullHandResults, self.debug)

        if "body" in self.modules:
            self.fullBodyResults = body.getBodyPoints(frame)

        self.currentFrame = outputFrame


    def render(self, frame, overlay = None):
        outputFrame = self.currentFrame.copy()
        if self.visibilityTime < 1_000:
            outputFrame = manipulation.overlayImage(outputFrame, frame, self.warpMatrix)
        # Resize to 1000 width, keeping aspect ratio
        outputFrame = cv2.resize(outputFrame, (1000, round(1000 * outputFrame.shape[0] / outputFrame.shape[1])))

        # Flip the frame horizontally
        if self.flip_horizontal:
            outputFrame = cv2.flip(outputFrame, 1)
        # Flip the frame vertically
        if self.flip_vertical:
            outputFrame = cv2.flip(outputFrame, 0)
        if overlay is not None:
            # Make overlay the same size as the output frame
            overlay = cv2.resize(overlay, (outputFrame.shape[1], outputFrame.shape[0]))
            # Create a mask of the overlay. Black pixels should be ignored
            mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
            # Make all coloured areas of the mask completely black on the main frame
            outputFrame = cv2.bitwise_and(outputFrame, outputFrame, mask=cv2.bitwise_not(mask))
            # Make transparent areas of the overlay completely black
            overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
            # For each channel, add the overlay to the main frame (where the mask is not black)
            for i in range(3):
                outputFrame[:, :, i] = outputFrame[:, :, i] + overlay[:, :, i]
        cv2.waitKey(1)
        cv2.imshow("Output", outputFrame)

    def kill(self):
        cv2.destroyAllWindows()
        screenspace.kill()
