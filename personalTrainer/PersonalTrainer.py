import time

import cv2
import numpy as np
import PoseTrackingModule as pm

video_capture = cv2.VideoCapture(0)

detector = pm.poseDetector()
prev_time = 0
count = 0
direction = 0

while True:
    _, img = video_capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (540, 720))
    img = detector.findPose(img, draw=False)
    lmlist = detector.findPosition(img, False)
    if lmlist:
        angle = detector.findAngle(img, 11, 13, 15, True)

        percentage = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (210, 310), (650, 100))
        color = (255, 0, 255)
        if percentage == 100:
            color = (0, 255, 0)
            if direction == 0:
                count += 0.5
                direction = 1
        if percentage == 0:
            color = (0, 255, 0)
            if direction == 1:
                count += 0.5
                direction = 0

        cv2.rectangle(img, (1100, 100), (1175, 650), (0, 255, 0), 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(percentage)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{count}', (70, 600), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS:{int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
