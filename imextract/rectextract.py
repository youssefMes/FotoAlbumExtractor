import numpy as np
import cv2

from matplotlib import pyplot as plt

def process_image(img):
    '''
    Parameters
    ----------
    img: ndarray - Image to extract

    Returns
    -------
    result: ndarray - Extracted image 
    '''
    img = detect_white_boarder(img)
    #detect_corner(img)
    #detect_conturs(img)

    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def detect_white_boarder(img, margin=40, threshold=180):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    y, x = gray.shape

    top = gray[0:margin, :]
    bottom = gray[y-margin:y, :]
    left = gray[:, 0:margin]
    right = gray[:, x-margin:x]

    top_color = cv2.calcHist([top], [0], None, [256], [0, 256]).argmax(axis=0)
    bottom_color = cv2.calcHist([bottom], [0], None, [256], [0, 256]).argmax(axis=0)
    left_color = cv2.calcHist([left], [0], None, [256], [0, 256]).argmax(axis=0)
    right_color = cv2.calcHist([right], [0], None, [256], [0, 256]).argmax(axis=0)

    print((top_color, bottom_color, left_color, right_color))

    if top_color > threshold and bottom_color > threshold and left_color > threshold and right_color > threshold:
        top_row = -1
        bottom_row = -1
        left_row = -1
        right_row = -1

        for i in range(0, margin - 2, 3):
            if np.any(top[i, margin:x-margin] < top_color ):
                top_row = i
            if np.any(bottom[i, margin:x-margin] < bottom_color ):
                bottom_row = i
            if np.any(left[margin:y-margin, i] < left_color ):
                left_row = i
            if np.any(right[margin:y-margin, i] < right_color ):
                right_row = i
        
        img = img[top_row:y - (bottom_row - 2), left_row:x - (right_row - 2) ]
    
    return img


def detect_corner(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 10, 15, 15)
    gray = cv2.Canny(gray, 20, 50)
    dst = cv2.cornerHarris(gray, 2, 5, 0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def detect_conturs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 10, 15, 15)
    gray = cv2.Canny(gray, 20, 50)

    im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    img = cv2.drawContours(img, contours, -1, (0,255,0), 1)

    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()