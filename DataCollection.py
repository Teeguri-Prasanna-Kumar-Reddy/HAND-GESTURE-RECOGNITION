import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgsize = 300

folder = "Data/3"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgsize,imgsize,3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset,x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        # Ensure that imgCrop has the same height as imgWhite
        if imgCrop.shape[0] != imgWhite.shape[0]:
            imgCrop = imgCrop[:imgWhite.shape[0], :]


        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgsize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgsize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgsize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        if imgCrop is not None and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{counter}.jpg',imgWhite)
        print(counter)