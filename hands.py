import cv2
import mediapipe as mp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
