import cv2
import imutils
from imutils import face_utils
import dlib

#img=cv2.imread('Face Shapes/Diamond/1.jpeg')
#cv2.imshow('img',img)
face_detector=dlib.get_frontal_face_detector()
landmarks_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

img_shapes=['Diamond','Oblong','Oval','Round','Square','Triangle']
train_data=[]
train_target=[]

def append_data(points,shape):

    face_dict={'Diamond':0,'Oblong':1,'Oval':2,'Round':3,'Square':4,'Triangle':5}
    
    my_points=points[2:9,0] #assgining point 3-point 9 to my points

    d1=my_points[6]-my_points[0]
    d2=my_points[6]-my_points[1]
    d3=my_points[6]-my_points[2]
    d4=my_points[6]-my_points[3]
    d5=my_points[6]-my_points[4]
    d6=my_points[6]-my_points[5]

    D1=d2/float(d1)*100
    D2=d3/float(d1)*100
    D3=d4/float(d1)*100
    D4=d5/float(d1)*100
    D5=d6/float(d1)*100

    train_data.append([D1,D2,D3,D4,D5])
    train_target.append(face_dict[shape])
    
for num in range(0,10):

    for shape in img_shapes:

        img=cv2.imread('Face Shapes/'+shape+'/'+str(num+1)+'.jpeg')
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        rect=face_detector(gray)
        points=landmarks_detector(gray,rect[0])  #receiving 68 points in face to points object
        points=face_utils.shape_to_np(points) #converting the points into a numpy array
##uncomment to see the photos and 68 l/marks
##        for p in points:
##
##            cv2.circle(img,(p[0],p[1]),2,(0,0,255),-1)
##        
##        cv2.imshow('LIVE',img)
##        cv2.waitKey(100)

        append_data(points,shape)

import pickle #can write arrays directly into files
import numpy as np

train_data=np.array(train_data)         #conveting into numpy arrays
train_target=np.array(train_target)

pickle.dump(train_data,open('train_data.pickle','wb'))
pickle.dump(train_target,open('train_target.pickle','wb'))

