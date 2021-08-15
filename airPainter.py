import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import handTrackingModule as htm


folderPath = "images"
listimg = os.listdir(folderPath)

overlay = []

for path in listimg:
    image = cv.imread(f'{folderPath}/{path}')
    overlay.append(image)

header = overlay[0]

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector()

xp = 0
yp = 0

imglayer = np.zeros((720, 1280, 3), np.uint8)

while True:

    success, img = cap.read()
    img = cv.flip(img, 1)

    # find hand landmarks

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


        fingers = detector.fingersUp()
        #print(fingers)


    # if selection mode
        if(fingers == [1, 1]):
            xp = 0
            yp = 0
            #print("selection")

            if y1 < 125:
                print(x1)
                if x1>70 and x1<258:
                    header = overlay[0]
                    color = (0, 0, 255)
                if x1>425 and x1<611:
                    header = overlay[1]
                    color = (255, 0, 0)
                if x1 > 786 and x1 < 973:
                    header = overlay[2]
                    color = (0, 255, 0)
                if x1 > 1080 and x1 < 1218:
                    header = overlay[3]
                    color = (0, 0, 0)

    # if drawing mode
        if(fingers == [1, 0]):

            print("")
            if(xp == 0 and yp == 0):
                xp = x1
                yp = y1

            if(color == (0, 0, 0)):
                cv.line(imglayer, (xp, yp), (x1, y1), color, 50, 3)
            cv.line(img, (xp, yp), (x1, y1), color, 10)
            cv.line(imglayer, (xp, yp), (x1, y1), color, 10)

            xp = x1
            yp = y1

    imgGray = cv.cvtColor(imglayer, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imglayer)

    img[0:125, 0:1280] = header




    #img = cv.addWeighted(img, 0.5, imglayer, 0.5, 0)
    cv.imshow("image", img)
    #cv.imshow("canvas", imglayer)
    #cv.imshow('inv', imgInv)
    cv.waitKey(1)

