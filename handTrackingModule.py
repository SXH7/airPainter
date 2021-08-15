import cv2 as cv
import mediapipe as mp
import time

class handDetector( ):
    def __init__(self, mode = False, max = 2, detectionConf = 0.5, trackingConf = 0.5):
        self.mode = mode
        self.max = max
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max, self.detectionConf, self.trackingConf)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        self.lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = (int(lm.x*w), int(lm.y * h))
                #print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                #if id == 0:
                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 0), cv.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []
        # Thumb
        '''if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)'''

        # Fingers
        for id in range(1, 3):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers







def main():
    ptime = 0
    ctime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        #print(lmList)

        '''if len(lmList) != 0:
            print(lmList[4])'''

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 225, 0), 3)

        cv.imshow('image', img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
