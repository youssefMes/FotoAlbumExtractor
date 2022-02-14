import cv2

def enhance_image(img, clipLimit):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return histogram_equalization(grayImg, clipLimit)

def histogram_equalization(img, clipLimit):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    return cv2.cvtColor(equalized,cv2.COLOR_GRAY2RGB)
