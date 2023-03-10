import cv2
import numpy as np


class OverlayOptions:
    STRETCH = 0


def generateWarpMatrix(sourceImage, newCorners, fitOption: OverlayOptions = OverlayOptions.STRETCH):
    # Create a matrix which maps the points of sourceImage to the points of newCorners
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
    # Create the matrix
    warpMatrix = cv2.getPerspectiveTransform(imageCorners, screenCorners)
    return warpMatrix


def findNewCoordinate(point, warpMatrix):
    # Applies the warp matrix to a point
    # By multiplying the point by the matrix, we can find the new position of the point
    point = np.array([point[0], point[1], 1])
    newPoint = np.matmul(warpMatrix, point)
    if newPoint[2] != 0:  # Prevent division by 0
        newPoint = newPoint / newPoint[2]
    return (newPoint[0], newPoint[1])


def warpImage(image, warpMatrix, dimensions, fitOption: OverlayOptions = OverlayOptions.STRETCH):
    """
    Takes an image and a warp matrix and returns the image warped to the new position
    """
    return cv2.warpPerspective(image, warpMatrix, (dimensions[1], dimensions[0]))


def overlayImage(base, overlay, warpMatrix, fitOption: OverlayOptions = OverlayOptions.STRETCH):
    warpedImage = warpImage(overlay, warpMatrix, base.shape[:2])
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
