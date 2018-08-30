import numpy as np
import cv2


def process_image(img, config):
    """
    Parameters
    ----------
    img: ndarray - Image to extract

    Returns
    -------
    result: ndarray - Extracted image 
    """
    img = detect_white_boarder(img)
    #detect_corner(img)
    #detect_conturs(img)

    return img


def margin(img, side="top", threshold=180, step_size=10, max_steps=10, min_percentage=0.4):
    height, width = img.shape

    frame = []

    margin = 0
    last = 0
    current = 0
    step = 0

    while current >= last and step < max_steps:
        margin += step_size

        if side == "top":
            frame = img[0:margin, :]
        elif side == "bottom":
            frame = img[height-margin:height, :]
        elif side == "left":
            frame = img[:, 0:margin]
        elif side == "right":
            frame = img[:, width-margin:width]

        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])

        y, x = frame.shape

        last = current
        current = np.sum(hist[threshold:]) / (y * x)

        if current < min_percentage:
            step += 1

    margin -= step_size
    color = hist.argmax(axis=0)

    if current < min_percentage:
        frame   = None
        margin  = 0
        color   = 0

    return frame, margin, color


def detect_white_boarder(img, threshold=180):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    top, top_margin, top_color = margin(gray, side="top", threshold=threshold)
    bottom, bottom_margin, bottom_color = margin(gray, side="bottom", threshold=threshold)
    left, left_margin, left_color = margin(gray, side="left", threshold=threshold)
    right, right_margin, right_color = margin(gray, side="right", threshold=threshold)

    top_row = 0
    bottom_row = 0
    left_row = 0
    right_row = 0

    for i in range(top_margin):
        if np.any(top[i, left_margin:width-right_margin] < top_color):
            top_row = i

    for i in range(bottom_margin):
        if np.any(bottom[i, left_margin:width-right_margin] < bottom_color):
            bottom_row = i
    
    for i in range(left_margin):
        if np.any(left[top_margin:height-bottom_margin, i] < left_color):
            left_row = i

    for i in range(right_margin):
        if np.any(right[top_margin:height-bottom_margin, i] < right_color):
            right_row = i
    
    img = img[top_row:height - bottom_row, left_row:width - right_row ]
    
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