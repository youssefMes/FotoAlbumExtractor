import numpy as np
import cv2
# import matplotlib.pyplot as plt


def process_image(img, config):
    """
    This extracts the image from a surrounding frame if a frame is existing.

    :param img: ndarray - The image to extract from frame.
    :param config: dictionary - The configuration of the config file.
    :return: Extracted image.
    :rtype ndarray.
    """
    max_window_size = config.get('ImageExtraction', 'MaxWindowSize')

    steps = config.get('ImageExtraction', 'Steps')
    offset = config.get('ImageExtraction', 'Offset')

    img = remove_boarder(img, steps, max_window_size, offset)

    return img


def remove_boarder(img, steps=25, max_window_size=0.1, gradient_offset=4):
    """
    Calculates the needed margins to remove the outlining boarder of the frame.

    :param img: ndarray - The image.
    :param steps: int - Each step increases the window where the best margin is found.
    :param max_window_size: float - The maximum size of the window to search the best margin
                            as percentage of the image.
    :param gradient_offset: int - The number of ignored gradients in the beginning of the search
                            to avoid to small margins.
    :return: Extracted image.
    :rtype ndarray.
    """
    height, width, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 20, 50)

    top_margin = find_margin(canny, "top", steps, max_window_size, gradient_offset)
    bottom_margin = find_margin(canny, "bottom", steps, max_window_size, gradient_offset)
    left_margin = find_margin(canny, "left", steps, max_window_size, gradient_offset)
    right_margin = find_margin(canny, "right", steps, max_window_size, gradient_offset)

    return img[top_margin:height - bottom_margin, left_margin:width - right_margin]


def find_margin(img, side="top", steps=25, max_frame_size=0.1, gradient_offset=4):
    """

    :param img:
    :param side:
    :param steps:
    :param max_frame_size:
    :param gradient_offset:
    :return:
    """

    height, width = img.shape

    # calculate the step size
    if side == "top" or side == "bottom":
        step_size = calc_step_size(img, steps, max_frame_size, axis=1)
    else:
        step_size = calc_step_size(img, steps, max_frame_size, axis=0)

    margin = 0
    step = 0

    results = []
    frame = []

    # get the feature for every search window
    while step < steps:
        margin += step_size

        if side == "top":
            frame = img[0:margin, :]
        elif side == "bottom":
            frame = img[height - margin:height, :]
        elif side == "left":
            frame = img[:, 0:margin]
        elif side == "right":
            frame = img[:, width - margin:width]

        results.append((frame > 0).sum())

        step += 1

    # calculate the gradients of the results array with the offset
    gradients = np.gradient(results)[gradient_offset:]

    # Plot the increasing feature count and the gradients

    # plt.title(side)
    # plt.plot(results)
    # plt.plot(np.gradient(results))
    # plt.show()

    # get the last iminimum not the first in the array
    last_min = len(gradients) - np.argmin(gradients[::-1])
    last_min += gradient_offset

    return last_min * step_size


def calc_step_size(img, steps, max_frame_size, axis=0):
    """

    :param img:
    :param steps:
    :param max_frame_size:
    :param axis:
    :return:
    """
    height, width = img.shape

    if axis == 1:
        step_size = np.ceil((width * max_frame_size) / steps)
    else:
        step_size = np.ceil((height * max_frame_size) / steps)

    return int(step_size)
