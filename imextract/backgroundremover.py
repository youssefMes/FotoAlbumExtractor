import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import time


def getPrimaryBackgroundColor(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    # get most occurring color
    background_color = hist.argmax(axis=0) 
    return background_color


def getBackgroundSpot(img, background_color, spot_size=200):
    spot_template = np.zeros((spot_size,spot_size,3), np.uint8)
    spot_template[:,:,0] = background_color
    spot_template[:,:,1] = background_color
    spot_template[:,:,2] = background_color

    # find big background spot
    method = cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(spot_template, img, method)
    # We want the minimum squared difference
    mn,_,location,_ = cv2.minMaxLoc(result)
    return location


def generateBinaryBackgroundImage(img, background_color, threshold=25):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,mask1 = cv2.threshold(gray,background_color + threshold,255,0)
    ret,mask2 = cv2.threshold(gray,background_color + threshold,255,0)
    combined = cv2.bitwise_not(cv2.bitwise_or(mask1, mask2))
    return combined


def separateBackground(binaryBackgroundImg, backgroundLocation):
    im_floodfill = binaryBackgroundImg.copy()
    h, w = binaryBackgroundImg.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, backgroundLocation, 128)
    im_floodfill[im_floodfill ==0] = 255
    im_floodfill[im_floodfill == 128] = 0

    return im_floodfill

  
def generateBinaryExtendedEdgeImage(img):
    # downsample for performance increase an noise reduction
    # temp = img
    temp = cv2.resize(img, (0,0), fx=0.45, fy=0.45)  
    temp = cv2.bilateralFilter(temp,15,75,75)  

    blur1 = cv2.GaussianBlur(temp,(3,3),0)
    blur2 = cv2.GaussianBlur(temp,(15,15),0)

    gradients = blur1 - blur2

    kernel = np.zeros((15,15),np.uint8)
    kernel = cv2.circle(kernel, (7,7), 5, 1, -1)

    gradients = cv2.morphologyEx(gradients, cv2.MORPH_CLOSE, kernel)

    binaryedge = cv2.resize(gradients, (img.shape[1],img.shape[0]))         

    return binaryedge


def checkForFeatures(inputImg, threshold = 10):
    blur1 = cv2.GaussianBlur(inputImg,(7,7),0)
    blur2 = cv2.GaussianBlur(inputImg,(15,15),0)
    gradients = blur1 - blur2

    pixelSum = np.sum(gradients[0:inputImg.shape[0]-1, 0:inputImg.shape[1]-1, 0:inputImg.shape[2]-1])
    average = pixelSum / (inputImg.shape[0] * inputImg.shape[1] * inputImg.shape[2])

    print(average)   

    return (average > threshold)


def cropImageRectangles(img, binaryBackgroundImage, minArea=-100, maxImageDimensionRelation=2.5, imagePadding=10):

    # initialize output images
    croppedImages = []

    im2, contours, hierarchy = cv2.findContours(binaryBackgroundImage,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    albumImageHeight = binaryBackgroundImage.shape[0]
    albumImageWidth = binaryBackgroundImage.shape[1]
    
    if(minArea < 0):
        minArea = albumImageHeight * albumImageWidth / (-minArea)

    countIngoredBecauseCornerDistance = 0
    countIngoredBecauseMinArea = 0
    countIngoredBecauseTotalAlbum = 0
    countIngoredBecauseDimensionRelation = 0

    for i in range(len(contours)):
        # the virtual corners correspond to the edges if every point should be in the image
        # the real corners are the contour points which are closest to the edge

        virtualcorners = [[albumImageWidth, albumImageHeight],[0, 0]]
        realcorners = [[albumImageWidth, albumImageHeight],[0, 0]]

        for j in range(len(contours[i])):
            if virtualcorners[0][0] > contours[i][j][0][0]:
                virtualcorners[0][0] = contours[i][j][0][0]
            if virtualcorners[0][1] > contours[i][j][0][1]:
                virtualcorners[0][1] = contours[i][j][0][1]
            if virtualcorners[1][0] < contours[i][j][0][0]:
                virtualcorners[1][0] = contours[i][j][0][0]
            if virtualcorners[1][1] < contours[i][j][0][1]:
                virtualcorners[1][1] = contours[i][j][0][1]

            if realcorners[0][0] + realcorners[0][1] > contours[i][j][0][0] + contours[i][j][0][1]:
                realcorners[0][0] = contours[i][j][0][0]
                realcorners[0][1] = contours[i][j][0][1]

            if realcorners[1][0] + realcorners[1][1] < contours[i][j][0][0] + contours[i][j][0][1]:
                realcorners[1][0] = contours[i][j][0][0]
                realcorners[1][1] = contours[i][j][0][1]

        # check if virtual corners are near real corners
        maxcornerdistance = math.sqrt(albumImageWidth*albumImageWidth + albumImageHeight*albumImageHeight)/20

        cornerdistance_topleft = math.sqrt(math.pow(realcorners[1][0] - virtualcorners[1][0] , 2) + math.pow(realcorners[1][1] - virtualcorners[1][1] , 2)) 
        cornerdistance_bottomright = math.sqrt(math.pow(realcorners[0][0] - virtualcorners[0][0] , 2) + math.pow(realcorners[0][1] - virtualcorners[0][1] , 2))   

        if cornerdistance_topleft > maxcornerdistance or cornerdistance_bottomright > maxcornerdistance:
            countIngoredBecauseCornerDistance += 1
            continue

        imageWidth = abs(realcorners[0][0] - realcorners[1][0])
        imageHeight = abs(realcorners[0][1] - realcorners[1][1])
        imageArea = abs(imageWidth * imageHeight)

        # dont save images that are the whole album image
        if img.shape[0] < imageHeight * 1.1 and img.shape[1] < imageWidth * 1.1:
            countIngoredBecauseTotalAlbum += 1
            continue

        # dont save images, that are to small
        if imageArea < minArea:
            countIngoredBecauseMinArea += 1
            continue

        # dont save images, that have weird dimensions
        if imageHeight/imageWidth > maxImageDimensionRelation or imageWidth/imageHeight > maxImageDimensionRelation: 
            countIngoredBecauseDimensionRelation += 1
            continue

        # if there is enough space add some padding
        if realcorners[0][1] - imagePadding > 0:
            realcorners[0][1] -= imagePadding
        if realcorners[0][0] - imagePadding > 0:
            realcorners[0][0] -= imagePadding
        if realcorners[1][1] + imagePadding < img.shape[0]:
            realcorners[1][1] += imagePadding
        if realcorners[1][0] + imagePadding < img.shape[1]:
            realcorners[1][0] += imagePadding

        crop = img[realcorners[0][1]:realcorners[1][1],realcorners[0][0]:realcorners[1][0]]
        croppedImages.append(crop)


    print("ignored due to CornerDistance: " + str(countIngoredBecauseCornerDistance))
    print("ignored due to MinArea: " + str(countIngoredBecauseMinArea))
    print("ignored due to TotalAlbum: " + str(countIngoredBecauseTotalAlbum))
    print("ignored due to DimensionRelation: " + str(countIngoredBecauseDimensionRelation))

    return croppedImages


def getImagesWithoutBackground(inputImg):
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(inputImg,-1,kernel)

    background_color = getPrimaryBackgroundColor(img)
    backgroundLocation = getBackgroundSpot(img, background_color)
    binaryImg = generateBinaryBackgroundImage(img, background_color)
    binaryBackgroundImg = separateBackground(binaryImg, backgroundLocation)

    binaryEdge = generateBinaryExtendedEdgeImage(inputImg)

    print(cv2.cvtColor(binaryEdge, cv2.COLOR_BGR2GRAY).shape)
    print(binaryBackgroundImg.shape)
    
    croppedImages = cropImageRectangles(img, binaryBackgroundImg)
    validCroppedImages = []

    for c in croppedImages:
        enoughFeatures = checkForFeatures(c)
        if enoughFeatures:
            validCroppedImages.append(c)

    return validCroppedImages


def processImage(img, name, outputpath="./output"):

    croppedImages = getImagesWithoutBackground(img)

    for i in range(len(croppedImages)):
        cv2.imwrite(outputpath + '/'+ name +'_' + str(i) + '.png', croppedImages[i])
