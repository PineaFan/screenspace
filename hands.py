import cv2
import mediapipe as mp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class HandModel:
    def __init__(self, handArray):
        self.thumb = handArray[0]
        self.index = handArray[1]
        self.middle = handArray[2]
        self.ring = handArray[3]
        self.pinky = handArray[4]

    def __eq__(self, other):
        return self.thumb == other.thumb and self.index == other.index and self.middle == other.middle and self.ring == other.ring and self.pinky == other.pinky


class Fist(HandModel):
    def __init__(self):
        super().__init__([False, False, False, False, False])


class Spread(HandModel):
    def __init__(self):
        super().__init__([True, True, True, True, True])


class Peace(HandModel):
    def __init__(self):
        super().__init__([True, False, False, False, False])


class IndexFinger(HandModel):
    def __init__(self):
        super().__init__([False, True, False, False, False])
        print(self.thumb, self.index, self.middle, self.ring, self.pinky)


class MiddleFinger(HandModel):
    def __init__(self):
        super().__init__([False, False, True, False, False])


class RingFinger(HandModel):
    def __init__(self):
        super().__init__([False, False, False, True, False])


class PinkyFinger(HandModel):
    def __init__(self):
        super().__init__([False, False, False, False, True])


mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.1
)
mpDraw = mp.solutions.drawing_utils

def renderHandPoints(frame, results, debug):
    if debug and results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(frame, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    return frame


def fromList(l, a):
    return [l[i] for i in a]


def getHandPoints(frame):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    landmarks = results.multi_hand_landmarks
    if landmarks:
        return landmarks, results
    return landmarks, results


def toVideospaceCoords(landmarks, width, height):
    # Landmarks are from [-1 to 1], with x y and z. We need to convert this to the screen space coordinates
    # of the camera, which is from [0 to width] and [0 to height]

    # We need to convert the x and y coordinates to the screen space coordinates
    # Z can be ignored

    output = []
    for landmark in landmarks:
        x = int((landmark.x) * width)
        y = int((landmark.y) * height)
        output.append((x, y))
    return output


import matplotlib.pyplot as plt

def getExtendedFingers(landmarks):
    # Each finger is written as points [1,2,3,4], [5,6,7,8] etc
    landmarks = landmarks.landmark
    fingers = [landmarks[x:x + 4] for x in range(1, 20, 4)]
    raised = [False for _ in range(5)]
    for finger in fingers:
        # Get the dot product of points 0 and 1 with 2 and 3
        # If the dot product is negative, the finger is extended

        # Get the vector from 0 to 1
        vector1 = [finger[1].x - finger[0].x, finger[1].y - finger[0].y, finger[1].z - finger[0].z]
        # Get the vector from 2 to 3
        vector2 = [finger[3].x - finger[2].x, finger[3].y - finger[2].y, finger[3].z - finger[2].z]
        # Get the dot product
        dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]
        # Set the variable
        raised[fingers.index(finger)] = dotProduct
    raised[0] = False
    return raised
