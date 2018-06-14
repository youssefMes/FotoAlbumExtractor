import cv2
import numpy as np

img = cv2.imread("gradients.png")
img2 = cv2.imread("binary-back.png")
print(img.shape, img2.shape)
img = cv2.bitwise_or(img, img2)

img = cv2.resize(img, (0,0), fx=0.1, fy=0.1) 

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


edges = gray
edges = cv2.Canny(gray,100,200,apertureSize = 3)
cv2.imshow('edges',edges)
cv2.waitKey(0)

# print(gray.shape)
# print(edges.shape)
temp = cv2.bitwise_or(gray, edges)

kernel = np.ones((2,18),np.uint8)
# kernel = cv2.circle(kernel, (7,7), 5, 1, -1)
dilatedImg = cv2.closing(temp,kernel,iterations = 1)

# minLineLength = 50
# maxLineGap = 5
# lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
# for x in range(0, len(lines)):
#     for x1,y1,x2,y2 in lines[x]:
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

lines = cv2.HoughLines(edges,1,np.pi/180,int(img.shape[1]/7))
print(len(lines))
for x in range(0, len(lines)):
    for rho,theta in lines[x]:
        deltaTheta = np.pi/36
        if theta > np.pi:
            theta = theta - np.pi
        
        if ((theta > np.pi - deltaTheta) or (theta < deltaTheta)) or ((theta > np.pi/2 - deltaTheta) and (theta < np.pi/2 + deltaTheta)):

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

img = cv2.bitwise_and(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),dilatedImg)

cv2.imshow('hough',img)
cv2.imshow('dilated',dilatedImg)
cv2.waitKey(0)