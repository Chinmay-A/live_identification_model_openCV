import pandas as pd
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

arr=[]
for i in range(10):
    arr.append(1)
for i in range(10,20):
    arr.append(0)

listd=[]

for i in range(20):
    listd.append(cv.imread(str(i)+".jpg"))

model=tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(80,80,3)),tf.keras.layers.Dense(128,activation=tf.nn.relu),tf.keras.layers.Dense(1,activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

print("fit data shape:"+ str(listd[0].shape))

model.fit(np.array(listd),np.array(arr),epochs=17)

camera=cv.VideoCapture(0)

while True:
    isTrue, frame=camera.read()

    altframe=frame

    frame=cv.resize(frame,(80,80))

    
    
    gframe=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    cv.imwrite("tempab.jpg",gframe)

    x=np.array([cv.imread("tempab.jpg")])

    print("input shape"+str(x.shape))

    res=model.predict(x)


    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    identobj=face_cascade.detectMultiScale(altframe,scaleFactor=1.1,minNeighbors=1)

    for (x,y,w,h) in identobj:
        cv.rectangle(altframe,(x,y),(x+w,y+h),(255,255,0),thickness=5)

    cv.putText(altframe,str(res),thickness=3,org=(50,50),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0))
    
    cv.imshow('Face Detection Video',altframe)

    #print(res)

    cv.waitKey(100)