from imutils.video import VideoStream
import imutils
import cv2
import mathutils
import numpy as np

# Start the video stream
vs = VideoStream(src=0).start()

screenspaceCorners = [(0, 0), (0, 0), (0, 0), (0, 0)]
defaultFullCodes = [screenspaceCorners for _ in range(4)]

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
arucoParams = cv2.aruco.DetectorParameters_create()

def getCurrentFrame():
    # Get the video stream from the webcam
    frame = vs.read()
    # frame = imutils.resize(frame, width=1000)
    return frame


def make_matrix(v1, v2, v3):
    a = v2-v1
    b = v3-v1

    c = a.cross(b)
    if c.magnitude>0:
        c = c.normalized()
    else:
        return None

    b2 = c.cross(a).normalized()
    a2 = a.normalized()
    m = mathutils.Matrix([a2, b2, c]).transposed()
    s = a.magnitude
    m *= mathutils.Matrix.Scale(s, 3)

    return m


def vectorFrom(p1, p2):
    return mathutils.Vector((p2[0] - p1[0], p2[1] - p1[1], 0))


def getScreenspacePoints(frame, videoFrame, debug, previousFullCodes) -> list[tuple[int, int]]:
    # Detect markers in the frame (Aruco 5x5 1000 0-3)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    confirmed = []  # Valid markers visible in the frame

    fullCodes = [x.copy() for x in defaultFullCodes]

    for listID, corner in enumerate(corners[:4]):
        for point in corner:  # for each corner in the list of markers
            index = 0
            for x, y in point:
                fullCodes[listID][index] = (x, y)
                if debug:
                    cv2.circle(videoFrame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Highlight it if debug is enabled
                if index == ids[listID][0]:  # If the point matches the position of the code
                    # E.G. If the marker is in the top left, this checks for the top left corner of the marker
                    screenspaceCorners[index] = (x, y)  # Save it for future frames (used if the marker is lost)
                    confirmed.append(index)
                index += 1

    if len(confirmed) == 3:
        (unknownPoint,) = {0, 1, 2, 3} - set(confirmed)

        v1 = vectorFrom(previousFullCodes[confirmed[0]][confirmed[0]], fullCodes[confirmed[0]][confirmed[0]])
        v2 = vectorFrom(previousFullCodes[confirmed[1]][confirmed[1]], fullCodes[confirmed[1]][confirmed[1]])
        v3 = vectorFrom(previousFullCodes[confirmed[2]][confirmed[2]], fullCodes[confirmed[2]][confirmed[2]])
        matrix = make_matrix(v1, v2, v3)
        if matrix:
            print(matrix)
            pointAsMatrix = mathutils.Matrix(([0, fullCodes[unknownPoint][unknownPoint][0]], [0, fullCodes[unknownPoint][unknownPoint][1]], [0, 0]))
            finalPoint = matrix @ pointAsMatrix
            # print(finalPoint)
            finalPoint 
            screenspaceCorners[unknownPoint] = (finalPoint[0], finalPoint[1])
            confirmed.append(unknownPoint)
            fullCodes[unknownPoint][unknownPoint] = finalPoint


    return screenspaceCorners, videoFrame, fullCodes


def addScreenspaceOverlay(frame, screenspaceCorners, debug=False):
    if (0, 0) not in screenspaceCorners:
        screenCorners = np.array([
            [screenspaceCorners[0][0], screenspaceCorners[0][1]],
            [screenspaceCorners[1][0], screenspaceCorners[1][1]],
            [screenspaceCorners[2][0], screenspaceCorners[2][1]],
            [screenspaceCorners[3][0], screenspaceCorners[3][1]]
        ], dtype="float32")

    if debug:
        for corner in screenspaceCorners:
            cv2.circle(frame, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)  # Highlight screen corners

        # Show a full polygon between the 4 corners
        cv2.line(frame, (int(screenspaceCorners[0][0]), int(screenspaceCorners[0][1])), (int(screenspaceCorners[1][0]), int(screenspaceCorners[1][1])), (0, 255, 0), 2)
        cv2.line(frame, (int(screenspaceCorners[1][0]), int(screenspaceCorners[1][1])), (int(screenspaceCorners[2][0]), int(screenspaceCorners[2][1])), (0, 255, 0), 2)
        cv2.line(frame, (int(screenspaceCorners[2][0]), int(screenspaceCorners[2][1])), (int(screenspaceCorners[3][0]), int(screenspaceCorners[3][1])), (0, 255, 0), 2)
        cv2.line(frame, (int(screenspaceCorners[3][0]), int(screenspaceCorners[3][1])), (int(screenspaceCorners[0][0]), int(screenspaceCorners[0][1])), (0, 255, 0), 2)

        # Show the diagonals
        cv2.line(frame, (int(screenspaceCorners[0][0]), int(screenspaceCorners[0][1])), (int(screenspaceCorners[2][0]), int(screenspaceCorners[2][1])), (0, 255, 0), 2)
        cv2.line(frame, (int(screenspaceCorners[1][0]), int(screenspaceCorners[1][1])), (int(screenspaceCorners[3][0]), int(screenspaceCorners[3][1])), (0, 255, 0), 2)
    return frame


def getMidpoints(screenspaceCorners, frame, debug):
    midpoints = (  # midpoint of 0&1, 1&2, 2&3, 3&0
        (int((screenspaceCorners[0][0] + screenspaceCorners[1][0]) / 2), int((screenspaceCorners[0][1] + screenspaceCorners[1][1]) / 2)),
        (int((screenspaceCorners[1][0] + screenspaceCorners[2][0]) / 2), int((screenspaceCorners[1][1] + screenspaceCorners[2][1]) / 2)),
        (int((screenspaceCorners[2][0] + screenspaceCorners[3][0]) / 2), int((screenspaceCorners[2][1] + screenspaceCorners[3][1]) / 2)),
        (int((screenspaceCorners[3][0] + screenspaceCorners[0][0]) / 2), int((screenspaceCorners[3][1] + screenspaceCorners[0][1]) / 2))
    )
    if debug:
        for point in midpoints:
            cv2.circle(frame, point, 5, (255, 0, 255), -1)
        # Show the diagonals
        cv2.line(frame, midpoints[0], midpoints[2], (255, 0, 255), 2)
        cv2.line(frame, midpoints[1], midpoints[3], (255, 0, 255), 2)
    return midpoints, frame


def kill():
    vs.stop()