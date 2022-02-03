import cv2
import application.extractor.remove as br
import application.extractor.rectangle_extract as re
import numpy as np

def get_background_extracted_images(img, config):
    """
    Approximately cuts out all the photos in the input image.

    :param img: the scanned site of an album.
    :param config: dictionary - The configuration of the config file.
    :returns: array of cut out images.
    :rtype: array
    """

    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)

    background_color = br.get_primary_background_color(img)

    spot_size = 200

    if not spot_size:
        sys.exit("Error in the config file!")

    background_location = br.get_background_spot(img, background_color, spot_size)

    binary_threshold = 25
    if not binary_threshold:
        sys.exit("Error in the config file!")

    binary_img = br.generate_binary_background_image(img, background_color, binary_threshold)
    binary_background_img = br.separate_background(binary_img, background_location)

    min_area = -100
    max_dimension_relation = 2.5
    image_padding = 10
    if not min_area or not max_dimension_relation or not image_padding:
        sys.exit("Error in the config file!")

    cropped_images = br.crop_image_rectangles(img, binary_background_img)

    feature_threshold = 10
    if not feature_threshold:
        sys.exit("Error in the config file!")

    valid_cropped_images = br.validate_cropped_images(cropped_images, feature_threshold)

    return valid_cropped_images

def get_frame_extracted_image(img, config):
    """
    This extracts the image from a surrounding frame if a frame is existing.

    :param img: ndarray - The image to extract from frame.
    :param config: dictionary - The configuration of the config file.
    :return: Extracted image.
    :rtype ndarray.
    """

    max_window_size = 0.1
    steps = 25
    offset = 4
    if not max_window_size or not steps or not offset:
        sys.exit("Error in the config file!")

    img = re.remove_border(img, steps, max_window_size, offset)

    return img
