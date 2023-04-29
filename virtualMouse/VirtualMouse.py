import time

import autopy
import cv2
import numpy as np

import HandTrackingModule as htm

fReduction = 100
threshold_distance = 35
smoothening = 7
wCam, hCam = 1280, 720

wScreen, hScreen = autopy.screen.size()
prev_time = 0
prev_locationX, prev_locationY = 0, 0
curr_locationX, curr_locationY = 0, 0

video_capture = cv2.VideoCapture(0)
video_capture.set(3, wCam)
video_capture.set(4, hCam)

detector = htm.handDetector(maxHands=1)

while True:
    _, img = video_capture.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    landmark_list, bbox = detector.findPosition(img, 0, False)

    if landmark_list:
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]

        fingers = detector.findFingersUp()
        cv2.rectangle(img, (fReduction, fReduction), (wCam - fReduction, hCam - fReduction), (255, 0, 255), 2)

        if fingers[1] and fingers[2] == 0:
            x3 = np.interp(x1, (fReduction, wCam - fReduction), (0, wScreen))
            y3 = np.interp(y1, (fReduction, hCam - fReduction), (0, hScreen))

            curr_locationX = prev_locationX + (x3 - prev_locationX) / smoothening
            curr_locationY = prev_locationY + (y3 - prev_locationY) / smoothening

            autopy.mouse.move(curr_locationX, curr_locationY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            prev_locationX, prev_locationY = curr_locationX, curr_locationY

        if fingers[1] and fingers[2]:
            distance, img, vertices = detector.findDistance(8, 12, img)
            if distance < threshold_distance:
                cv2.circle(img, (vertices[-2], vertices[-1]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS:{int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
