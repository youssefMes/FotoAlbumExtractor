import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import sys
import csv
import time
import configparser
import math

# inputfolder = str(sys.argv[1])

# data storing path
frontalFacesClassifier = cv2.CascadeClassifier(sys.exec_prefix + "/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
profilesClassifier = cv2.CascadeClassifier(sys.exec_prefix + "/Lib/site-packages/cv2/data/haarcascade_profileface.xml")
# input images path
imagePath = "../output/"

# output images path
facesPath = "./output/"
# extracted faces
singleFacesPath = "./output/faces/"

if not os.path.exists(facesPath):
    os.makedirs(facesPath)
    os.makedirs(singleFacesPath)

# parse config file
config = configparser.ConfigParser()
config.read('../config')
scaleFactor = config.getfloat('FaceDetection', 'ScaleFactor')
minNeighbors = config.getint('FaceDetection', 'Neighbors')

# count detected faces
numberFaces = 0
numberProfiles = 0

# array for coordinates
coordinates = ["image, x pos., y pos., width, height:"]

# process images 
for file in os.listdir(imagePath):
    filename = os.fsdecode(file)
    if filename.endswith(".png"): 
        imgname = filename.partition(".")[0]
        img = cv2.imread(imagePath + filename)
        # convert the image to gray image as opencv face detector expects gray images 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # frontal faces
        frontalFaces = frontalFacesClassifier.detectMultiScale(img_gray, scaleFactor, minNeighbors) 
        # profiles
        profiles = profilesClassifier.detectMultiScale(img_gray, scaleFactor, minNeighbors) 

        listFrontalFaces = list(frontalFaces)
        listProfiles = list(profiles)

        for x, y, w, h in listFrontalFaces:
            for i, profile in enumerate(listProfiles):
                x1 = profile[0]
                y1 = profile[1]
                w1 = profile[2]
                h1 = profile[3]
                if float(abs(y1-y)) < h1/3 and float(abs(x1-x)) < w1/3:
                    del listProfiles[i]

        listFrontalFaces.extend(listProfiles)

        # save single faces
        if listFrontalFaces or listProfiles:
            i = 0
            for (x2, y2, w2, h2) in listFrontalFaces:
                sub_face = img[y2:y2+h2, x2:x2+w2]
                cv2.imwrite(singleFacesPath + str(imgname) + "_" + str(i) + ".png", sub_face)
                coordinates.append([imgname, x2, y2, w2, h2])
                i += 1

            # mark single faces in original image (green and yellow)
            for (x3, y3, w3, h3) in listFrontalFaces:
                cv2.rectangle(img, (x3, y3), (x3+w3, y3+h3), (0, 255, 0), 2)

        cv2.imwrite(facesPath + filename, img)

        sys.stdout.write("-")
        sys.stdout.flush()

coordinates.insert(1, "# faces: " + str(numberFaces))
coordinates.insert(2, "# profiles: " + str(numberProfiles))

with open(facesPath + "coordinates.csv", "w") as outfile:
    for entries in coordinates:
        outfile.write(str(entries))
        outfile.write("\n") 

sys.stdout.write("\n")

