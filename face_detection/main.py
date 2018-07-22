import cv2
from matplotlib import pyplot as plt
import os
import sys
import time
import configparser

# input path
imagePath = "../background_detection/output1/"
# imagePath = "../background_detection/output2/"
# training data storing path
haar_face_cascade = cv2.CascadeClassifier(sys.exec_prefix + '/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# output path
facesPath = "./output1/"
# facesPath = "./output2/"

if not os.path.exists(facesPath):
    os.makedirs(facesPath)

# parse config file
config = configparser.ConfigParser()
config.read('../config.ini')
scaleFactor = config.getfloat('FaceDetection', 'ScaleFactor')
minNeighbors = config.getint('FaceDetection', 'Neighbors')

# scaleFactor = config['FaceDetection']['ScaleFactor']
# print(scaleFactor)

for file in os.listdir(imagePath):
    filename = os.fsdecode(file)
    # print(filename)
    if filename.endswith(".png"): 
        img = cv2.imread(imagePath+filename)
        #convert the test image to gray image as opencv face detector expects gray images 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # print(img_gray.dtype)

        faces = haar_face_cascade.detectMultiScale(img_gray, scaleFactor, minNeighbors);  
 
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # plt.imshow(img)
        # plt.show()

        cv2.imwrite(facesPath + filename, img)

        sys.stdout.write("-")
        sys.stdout.flush()

        continue
    # else:
        continue

sys.stdout.write("\n")
