import cv2
import numpy as np

img = cv2.imread("gradients.png")
img2 = cv2.imread("binary-back.png")
img = cv2.bitwise_or(img, img2)

img = cv2.resize(img, (0,0), fx=0.1, fy=0.1) 

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


edges = gray
edges = cv2.Canny(gray,100,200,apertureSize = 3)
cv2.imshow('edges',edges)
cv2.waitKey(0)

# minLineLength = 50
# maxLineGap = 5
# lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
# for x in range(0, len(lines)):
#     for x1,y1,x2,y2 in lines[x]:
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

lines = cv2.HoughLines(edges,1,np.pi/180,int(img.shape[1]/5))
print(len(lines))
for x in range(0, len(lines)):
    for rho,theta in lines[x]:
        deltaTheta = np.pi/36
        if theta > np.pi:
            theta = theta - np.pi
        
        # print(theta)

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

cv2.imshow('hough',img)
cv2.waitKey(0)