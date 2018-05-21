import numpy as np
import cv2
import matplotlib.pyplot as plt

class BackgroundRemover:
    @staticmethod
    def getPrimaryBackgroundColor(self, img):
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        # get most occurring color
        background_color = hist.argmax(axis=0) 
        return background_color

    @staticmethod
    def getBackgroundSpot(self, img, background_color, spot_size=50):
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

    @staticmethod
    def generateBinaryBackgroundImage(self, img, background_color, threshold=7):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,mask1 = cv2.threshold(gray,background_color + threshold,255,0)
        ret,mask2 = cv2.threshold(gray,background_color + threshold,255,0)
        combined = cv2.bitwise_not(cv2.bitwise_or(mask1, mask1))
        return combined

    @staticmethod
    def separateBackground(self, binaryBackgroundImg, backgroundLocation):
        im_floodfill = binaryBackgroundImg.copy()
        h, w = binaryBackgroundImg.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, backgroundLocation, 128)
        im_floodfill[im_floodfill ==0] = 255
        im_floodfill[im_floodfill == 128] = 0

        return im_floodfill

    @staticmethod
    def cropImageRectangles(self, binaryBackgroundImage, minArea=100000, maxImageDimensionRelation=2.5):
        # initialize output images
        croppedImages = []

        im2, contours, hierarchy = cv2.findContours(binaryBackgroundImage,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        albumImageHeight = binaryBackgroundImage.shape[0]
        albumImageWidth = binaryBackgroundImage.shape[1]

        for i in range(len(contours)):
            corners = [[albumImageWidth, albumImageHeight],[0, 0]]
            for j in range(len(contours[i])):
                if corners[0][0] > contours[i][j][0][0]:
                    corners[0][0] = contours[i][j][0][0]
                if corners[0][1] > contours[i][j][0][1]:
                    corners[0][1] = contours[i][j][0][1]
                if corners[1][0] < contours[i][j][0][0]:
                    corners[1][0] = contours[i][j][0][0]
                if corners[1][1] < contours[i][j][0][1]:
                    corners[1][1] = contours[i][j][0][1]
                
            imageWidth = corners[0][0] - corners[1][0]
            imageHeight = corners[0][1] - corners[1][1]
            imageArea = abs(imageWidth * imageHeight)
            if(imageArea >= minArea and imageHeight/imageWidth < maxImageDimensionRelation and imageWidth/imageHeight < maxImageDimensionRelation): 
                crop = img[corners[0][1]:corners[1][1],corners[0][0]:corners[1][0]]
                croppedImages.append(crop)
                cv2.imwrite('image-' + str(len(croppedImages)) + '.png',crop)
                

        return croppedImages

# Load an color image in grayscale
inputImg = cv2.imread('../input/A1/21.tif')

# img = cv2.resize(inputImg, (0,0), fx=0.3, fy=0.3) 
kernel = np.ones((5,5),np.float32)/25
img = cv2.filter2D(inputImg,-1,kernel)

background_color = BackgroundRemover.getPrimaryBackgroundColor(None,img)

backgroundLocation = BackgroundRemover.getBackgroundSpot(None,img, background_color)

binaryImg = BackgroundRemover.generateBinaryBackgroundImage(None,img, background_color)

binaryBackgroundImg = BackgroundRemover.separateBackground(None,binaryImg, backgroundLocation)

croppedImages = BackgroundRemover.cropImageRectangles(None,binaryBackgroundImg)

