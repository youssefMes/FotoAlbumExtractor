import cv2

def enhance_image(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return histogram_equalization(grayImg)

def histogram_equalization(img):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    return cv2.cvtColor(equalized,cv2.COLOR_GRAY2RGB)
