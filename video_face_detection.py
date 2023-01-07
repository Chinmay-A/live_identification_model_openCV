import cv2 as cv
from matplotlib import pyplot as plt
import time as time

video=cv.VideoCapture(0)



while True:
    isTrue,frame=video.read()

    gframe=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    identobj=face_cascade.detectMultiScale(gframe,scaleFactor=1.1,minNeighbors=1)

    for (x,y,w,h) in identobj:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),thickness=10)
    
    cv.imshow('Face Detection Video',frame)

    cv.waitKey(1)