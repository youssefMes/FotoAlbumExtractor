import numpy as np
import cv2
import math


def get_primary_background_color(img):
    """
    Returns the primarily used color in the images, which is assumed to be the background color.

    :param img: this is the image
    :returns the primary hue tone.
    :rtype int
    """

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # get most occurring color
    background_color = hist.argmax(axis=0)

    return background_color


def get_background_spot(img, background_color, spot_size=200):
    """
    Returns a position in the image, which is the most similar spot to the background color.

    :param img: this is the image
    :param background_color: this is the background color
    :param spot_size:   the size of the searched spot.
                        The higher the value, the slower the search and up to a certain size more stable
    :returns x, y coordinate of the background spot.
    :rtype tuple
    """

    spot_template = np.zeros((spot_size, spot_size, 3), np.uint8)
    spot_template[:, :, 0] = background_color
    spot_template[:, :, 1] = background_color
    spot_template[:, :, 2] = background_color

    # find big background spot
    method = cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(spot_template, img, method)
    # We want the minimum squared difference
    mn, _, location, _ = cv2.minMaxLoc(result)

    return location


def generate_binary_background_image(img, background_color, threshold=25):
    """
    Returns a binary image, which where the background color with some threshold is separated from the rest.

    :param img: this is the image
    :param background_color: this is the background color
    :param threshold: the threshold around the primary background color, which still should belong to the background.
    :returns: binary image. 
    :rtype: array  
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask1 = cv2.threshold(gray, background_color + threshold, 255, 0)
    ret, mask2 = cv2.threshold(gray, background_color + threshold, 255, 0)
    combined = cv2.bitwise_not(cv2.bitwise_or(mask1, mask2))

    return combined


def separate_background(binary_background_img, background_location):
    """
    Returns a binary image, where the background ist black and the image locations are white.

    :param binary_background_img: binary version of the image
    :param background_location: a location (x,y) where there is some background
    :returns: binary image. 
    :rtype: array  
    """

    im_floodfill = binary_background_img.copy()
    h, w = binary_background_img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(im_floodfill, mask, background_location, 128)

    im_floodfill[im_floodfill == 0] = 255
    im_floodfill[im_floodfill == 128] = 0

    return im_floodfill

  
def generate_binary_extended_edge_image(img):
    """
    Returns a binary image, at which all the edges are white.
    For a better performance and noise reduction the image is downsampled and bilateral filter.
    The edges are detected by a comparision of the original image and a blurred one.
    By this small gaps in the edges are also detected.

    :param img: binary version of the image
    :returns: binary image where the edges are white spoches and the rest is black. 
    :rtype: array  
    """

    temp = cv2.resize(img, (0, 0), fx=0.45, fy=0.45)
    temp = cv2.bilateralFilter(temp, 15, 75, 75)

    blur1 = cv2.GaussianBlur(temp, (3, 3), 0)
    blur2 = cv2.GaussianBlur(temp, (15, 15), 0)

    gradients = blur1 - blur2

    kernel = np.zeros((15, 15), np.uint8)
    kernel = cv2.circle(kernel, (7, 7), 5, 1, -1)

    gradients = cv2.morphologyEx(gradients, cv2.MORPH_CLOSE, kernel)

    return cv2.resize(gradients, (img.shape[1], img.shape[0]))


def check_for_features(img, threshold=10):
    """
    Returns true or false dependent on the amount of features (corners and edges) which are in the image.
    Used to remove images without content (only background).

    :param img: input image
    :param threshold: the necessary amount of features needed to be regarded as image
    :returns: boolean, if image as enough features 
    :rtype: bool  
    """

    blur1 = cv2.GaussianBlur(img, (7, 7), 0)
    blur2 = cv2.GaussianBlur(img, (15, 15), 0)
    gradients = blur1 - blur2

    pixel_sum = np.sum(gradients[0:img.shape[0]-1, 0:img.shape[1]-1, 0:img.shape[2]-1])
    average = pixel_sum / (img.shape[0] * img.shape[1] * img.shape[2])

    return average > threshold


def crop_image_rectangles(img, binary_background_image, min_area=-100, max_dimension_relation=2.5, image_padding=10):
    """
    Returns an array of images, which are cut out of the original image.
    The cut is based on the binary background image.
    During this process unrelevant (to small, to monoton, ...) images are sorted out.

    :param img: input image
    :param binary_background_image: binary image showing where background and where foreground is.
    :param min_area: the size(area) an image must at least have to be considered as an image.
    :param max_dimension_relation: the maximum relation between the width and the height of an image
                                    (-> strips are not allowed)
    :param image_padding: the padding with which image is cut out of the original photo.
    :returns: an array of all the images in the scan
    :rtype: array  
    """

    # initialize output images
    cropped_images = []

    im2, contours, hierarchy = cv2.findContours(binary_background_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    album_image_height = binary_background_image.shape[0]
    album_image_width = binary_background_image.shape[1]
    
    if min_area < 0:
        min_area = album_image_height * album_image_width / (-min_area)

    count_ignored_because_corner_distance = 0
    count_ignored_because_min_area = 0
    count_ignored_because_total_album = 0
    count_ignored_because_dimension_relation = 0

    for i in range(len(contours)):
        # the virtual corners correspond to the edges if every point should be in the image
        # the real corners are the contour points which are closest to the edge

        virtual_corners = [[album_image_width, album_image_height], [0, 0]]
        real_corners = [[album_image_width, album_image_height], [0, 0]]

        for j in range(len(contours[i])):
            if virtual_corners[0][0] > contours[i][j][0][0]:
                virtual_corners[0][0] = contours[i][j][0][0]
            if virtual_corners[0][1] > contours[i][j][0][1]:
                virtual_corners[0][1] = contours[i][j][0][1]
            if virtual_corners[1][0] < contours[i][j][0][0]:
                virtual_corners[1][0] = contours[i][j][0][0]
            if virtual_corners[1][1] < contours[i][j][0][1]:
                virtual_corners[1][1] = contours[i][j][0][1]

            if real_corners[0][0] + real_corners[0][1] > contours[i][j][0][0] + contours[i][j][0][1]:
                real_corners[0][0] = contours[i][j][0][0]
                real_corners[0][1] = contours[i][j][0][1]

            if real_corners[1][0] + real_corners[1][1] < contours[i][j][0][0] + contours[i][j][0][1]:
                real_corners[1][0] = contours[i][j][0][0]
                real_corners[1][1] = contours[i][j][0][1]

        # check if virtual corners are near real corners
        max_corner_distance = math.sqrt(album_image_width*album_image_width
                                        + album_image_height*album_image_height) / 20

        corner_distance_topleft = math.sqrt(math.pow(real_corners[1][0] - virtual_corners[1][0], 2)
                                            + math.pow(real_corners[1][1] - virtual_corners[1][1], 2))

        corner_distance_bottomright = math.sqrt(math.pow(real_corners[0][0] - virtual_corners[0][0], 2)
                                                + math.pow(real_corners[0][1] - virtual_corners[0][1], 2))

        if corner_distance_topleft > max_corner_distance or corner_distance_bottomright > max_corner_distance:
            count_ignored_because_corner_distance += 1
            continue

        image_width = abs(real_corners[0][0] - real_corners[1][0])
        image_height = abs(real_corners[0][1] - real_corners[1][1])
        image_area = abs(image_width * image_height)

        # dont save images that are the whole album image
        if img.shape[0] < image_height * 1.1 and img.shape[1] < image_width * 1.1:
            count_ignored_because_total_album += 1
            continue

        # dont save images, that are to small
        if image_area < min_area:
            count_ignored_because_min_area += 1
            continue

        # dont save images, that have weird dimensions
        if image_height/image_width > max_dimension_relation or image_width/image_height > max_dimension_relation:
            count_ignored_because_dimension_relation += 1
            continue

        # if there is enough space add some padding
        if real_corners[0][1] - image_padding > 0:
            real_corners[0][1] -= image_padding
        if real_corners[0][0] - image_padding > 0:
            real_corners[0][0] -= image_padding
        if real_corners[1][1] + image_padding < img.shape[0]:
            real_corners[1][1] += image_padding
        if real_corners[1][0] + image_padding < img.shape[1]:
            real_corners[1][0] += image_padding

        crop = img[real_corners[0][1]:real_corners[1][1],real_corners[0][0]:real_corners[1][0]]
        cropped_images.append(crop)

    return cropped_images


def validate_cropped_images(cropped_images, feature_threshold):
    """
    Validated the cropped image by checking for feature.

    :param feature_threshold: the necessary amount of features needed to be regarded as image
    :param cropped_images: array - An array of cropped images
    :return: An array of validated cropped images
    :rtype array
    """
    valid_cropped_images = []

    for image in cropped_images:
        enough_features = check_for_features(image, feature_threshold)
        if enough_features:
            valid_cropped_images.append(image)

    return valid_cropped_images
