import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

frames_list=[]


video=cv.VideoCapture(0)

for i in range(10,20):
    
    isTrue,frame=video.read()

    frame=cv.resize(frame,(80,80))
    
    gframe=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    cv.imshow('currentframe',frame)

    cv.imwrite(str(i)+'.jpg',gframe)

    cv.waitKey(0)


# for i in range(10,20):
    
#     isTrue,frame=video.read()

#     frame=cv.resize(frame,(80,80))
    
#     gframe=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

#     cv.imshow('currentframe',frame)

#     cv.imwrite(str(i)+'.jpg',gframe)

#     cv.waitKey(0)





