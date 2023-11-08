import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgsize = 300


labels = ["1","2","3"]


while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)

        else:
            k = imgsize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgsize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x - offset + 90, y - offset -50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

        if imgCrop is not None and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image",imgOutput)
    cv2.waitKey(1)