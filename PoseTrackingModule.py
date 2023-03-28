import time

import cv2 as cv
import mediapipe as mp


class poseDetector():

    def __init__(self, mode=False, upBody=False, complexity=1, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.complexity = complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.complexity, self.smooth, self.detectionCon,
                                     self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        if results.pose_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
        # for id, lm in enumerate(results.pose_landmarks.landmark):
        #     h, w, c = img.shape
        #     cx, cy = int(lm.x * w), int(lm.y * h)
        #     cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)


def main():
    cap = cv.VideoCapture("./Videos/vid1.mp4")
    pTime = 0
    img = detector = poseDetector()

    while True:
        success, img = cap.read()
        detector.findPose(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv.imshow("Image", img)

        cv.waitKey(1)


if __name__ == "main":
    main()
