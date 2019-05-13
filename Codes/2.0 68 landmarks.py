import cv2 #cv2, opencv for image processing tasks

camera=cv2.VideoCapture(0)  #loading the default(0) camera into camera object

import dlib

face_detector=dlib.get_frontal_face_detector()
#loading a face detecting classifer from dlib library into face_detector object

landmarks_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#loading the 68 landmarks detecting classifier into landmarks_detector

import imutils #image utilities library
from imutils import face_utils

while(True):

    ret,img=camera.read()       #reading one frame from camera and assign it to img  
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting the rgb image into a gray image 

    rect=face_detector(gray)    #sending the gray image into face_detector Algorithm
                                #rect contains x1,y1,x2,y2 points of faces
##    x1=rect[0].left()
##    y1=rect[0].top()
##    x2=rect[0].right()
##    y2=rect[0].bottom()
##
##    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

    points=landmarks_detector(gray,rect[0])  #receiving 68 points in face to points object
    points=face_utils.shape_to_np(points) #converting the points into a numpy array

    for p in points:

        cv2.circle(img,(p[0],p[1]),2,(0,0,255),-1)
        
    
    cv2.imshow('LIVE',img)
    cv2.waitKey(1)          #1ms delay

