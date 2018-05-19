import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load an color image in grayscale
inputImg = cv2.imread('../input/A1/21.tif')

img = cv2.resize(inputImg, (0,0), fx=0.3, fy=0.3) 
kernel = np.ones((5,5),np.float32)/25
img = cv2.filter2D(img,-1,kernel)

hist = cv2.calcHist([img],[0],None,[256],[0,256])

# get most occurring color
background_color = hist.argmax(axis=0)

blank_image = np.zeros((55,55,3), np.uint8)
blank_image[:,:,0] = background_color
blank_image[:,:,1] = background_color
blank_image[:,:,2] = background_color

# find big background spot
method = cv2.TM_SQDIFF_NORMED
result = cv2.matchTemplate(blank_image, img, method)
# We want the minimum squared difference
mn,_,mnLoc,_ = cv2.minMaxLoc(result)
MPx,MPy = mnLoc
trows,tcols = blank_image.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,3,1)
ret,mask1 = cv2.threshold(gray,background_color+7,255,0)
# first = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

ret,mask2 = cv2.threshold(gray,background_color-7,255,0)
# second = cv2.findContours(thresh2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

combined = cv2.bitwise_not(cv2.bitwise_or(mask1, mask1))
# print(_)
mask = np.zeros(img.shape[:-1],np.uint8)
# cv2.drawContours(mask,contours,-1,(255,255,255),-1)
# mask = np.zeros(img.shape[:-1],np.uint8)
# cv2.floodFill(mask,None,mnLoc,(255,0,0))

# cv2.rectangle(img, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)
    

# plt.plot(hist)
# plt.show()
cv2.imshow('image',combined)
cv2.waitKey(0)
cv2.destroyAllWindows()