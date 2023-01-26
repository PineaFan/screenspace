import numpy as np
import cv2


class OverlayOptions:
    STRETCH = 0


def generateWarpMatrix(sourceImage, newCorners, fitOption: OverlayOptions = OverlayOptions.STRETCH):
    imageCorners = np.array([
        [0, 0],
        [sourceImage.shape[1], 0],
        [sourceImage.shape[1], sourceImage.shape[0]],
        [0, sourceImage.shape[0]]
    ], dtype="float32")
    screenCorners = np.array([
        [newCorners[0][0], newCorners[0][1]],
        [newCorners[1][0], newCorners[1][1]],
        [newCorners[2][0], newCorners[2][1]],
        [newCorners[3][0], newCorners[3][1]]
    ], dtype="float32")
    warpMatrix = cv2.getPerspectiveTransform(imageCorners, screenCorners)
    return warpMatrix


def findNewCoordinate(point, warpMatrix):
    # Applies the warp matrix to a point
    # This is used to find the new coordinates of a point in the new image

    point = np.array([point[0], point[1], 1])
    newPoint = np.matmul(warpMatrix, point)
    if newPoint[2] != 0:
        newPoint = newPoint / newPoint[2]
    return (newPoint[0], newPoint[1])


def warpImage(image, warpMatrix, dimensions, fitOption: OverlayOptions = OverlayOptions.STRETCH):
    return cv2.warpPerspective(image, warpMatrix, (dimensions[1], dimensions[0]))


def overlayImage(base, overlay, warpMatrix, fitOption: OverlayOptions = OverlayOptions.STRETCH):
    warpedImage = warpImage(overlay, warpMatrix, base.shape[:2])
    # This relies on the background being pure black. If it isn't, this will not work

    # Generate a mask by making every pixel is not transparent pure black
    # This does not work with translucent images
    mask = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

    # Apply the mask to the background image
    base = cv2.bitwise_and(base, base, mask=cv2.bitwise_not(mask))
    # Then add the overlay to the background
    base = cv2.addWeighted(base, 1, warpedImage, 1, 0)

    return base


def createImageWithDimensions(width, height, colour=(0, 0, 0)):
    colour = colour + (1,)
    return np.zeros((height, width, 3, 1), np.uint8) + colour
