"""
This is an example project - It doesn't work well, but shows what the driver can do, and how the hand tracking works
"""

import cv2
import numpy as np
from driver import Driver

maxBounceAngle = 60
maxBallSpeed = 10

driver = Driver(debug=False, modules=["hands"])
background = np.zeros((150, 300, 3), np.uint8)  # Create a white 150x300 image for the background
background[:] = (255, 255, 255)

debug = False


class Ball:
    def __init__(self):
        self.x = background.shape[1] // 2
        self.y = background.shape[0] // 2
        self.vx = 0
        self.vy = 0
        self.speed = 3
        self.radius = 2
        self.collisionEnabled = True

    def update(self, paddles):
        if self.y - self.radius < 0 or self.y + self.radius> background.shape[0]:
            self.vy *= -1
        if self.x - self.radius < 0 or self.x + self.radius > background.shape[1] and self.collisionEnabled:
            self.speed //= 2
            self.speed = max(self.speed, 2)
            self.vx *= -1
            self.collisionEnabled = False
        for paddle in paddles:
            # Check if the ball is colliding with the paddle
            if self.x + self.radius > paddle.corners[0][0] and \
                self.x - self.radius < paddle.corners[1][0] and \
                self.y + self.radius > paddle.corners[0][1] and \
                self.y - self.radius < paddle.corners[1][1] and \
                self.collisionEnabled:
                relativeIntersectY = (paddle.y + (paddle.height // 2)) - self.y
                normalizedRelativeIntersectionY = (relativeIntersectY / paddle.height)
                bounceAngle = normalizedRelativeIntersectionY * maxBounceAngle
                print(relativeIntersectY, normalizedRelativeIntersectionY, bounceAngle)
                self.vx = np.sin(bounceAngle)
                self.vy = np.cos(bounceAngle)
                self.collisionEnabled = False
                self.speed = min(max(self.speed + 1, 2), maxBallSpeed)
        if self.x + self.radius > paddles[0].x + paddles[0].width and self.x + self.radius < paddles[1].x:
            self.collisionEnabled = True
        self.x += round(self.vx * self.speed)
        self.y += round(self.vy * self.speed)

    def start(self):
        self.vx = 1
        self.vy = 1


class Paddle:
    def __init__(self, side = "left"):
        self.left = side == "left"
        self.y = background.shape[0] // 2
        self.width = 4
        self.x = 5 if self.left else background.shape[1] - 5 - self.width
        self.height = 30

    def update(self, y):
        self.y = y

    @property
    def corners(self):
        return (self.x, self.y - (self.height // 2)), (self.x + self.width, self.y + (self.height // 2))


ball = None
paddles = [Paddle("left"), Paddle("right")]

x = 0
while True:
    frame = background.copy()
    driver.calculate(background.shape[1], background.shape[0])
    if not ball:
        ball = Ball()
        ball.start()
    # Show point 8 from the screenspace hand points
    if driver.screenspaceHandPoints:
        for i, hand in enumerate(driver.screenspaceHandPoints):
            point = hand[8]
            if i < len(paddles):
                paddles[i].update(round(point[1] - paddles[i].height // 2))
                cv2.rectangle(frame, *paddles[i].corners, (0, 0, 255), -1)

    cv2.circle(frame, (ball.x, ball.y), ball.radius, (0, 0, 255), -1)

    if debug:
        frame = cv2.resize(frame, (1500, 1000))
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    else:
        driver.render(frame)
    ball.update(paddles)
