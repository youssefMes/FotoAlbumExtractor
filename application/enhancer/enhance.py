import cv2

def enhance_image(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #return conversion_to_YCbCr(grayImg)
    return histogram_equalization(grayImg)

def histogram_equalization(img):
    #image = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    return cv2.cvtColor(equalized,cv2.COLOR_GRAY2RGB)

def conversion_to_YCbCr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
