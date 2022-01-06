import numpy as np
import cv2


def remove_border(img, steps=25, max_window_size=0.1, gradient_offset=4):
    """
    Calculates the needed margins to remove the outlining border of the image.

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


def find_margin(canny, side="top", steps=25, max_frame_size=0.1, gradient_offset=4):
    """
    This method creates a window on the given side. The window is increased step by step with the size calculated by the
    maximum frame size and the number of steps. For each window the number of white pixel is counted and saved.
    In the end the gradients between those counted pixel is calculated and the smallest gradient is used to find the
    following maximum. This maximum gives the steps needed for the margin.

    :param canny: ndarray - Image created by canny operation
    :param side: string - ["top", "bottom", "left", "right"] The side where the frame is searched
    :param steps: int - The steps used to find the frame
    :param max_frame_size: float - The maximal window size
    :param gradient_offset: int - The number of ignored steps to prevent finding the already existing boarder
    :return: The margin of the given side
    :rtype int
    """

    height, width = canny.shape

    # calculate the step size
    if side == "top" or side == "bottom":
        step_size = calc_step_size(canny, steps, max_frame_size, axis=1)
    else:
        step_size = calc_step_size(canny, steps, max_frame_size, axis=0)

    margin = 0
    step = 0

    results = []
    frame = []

    # get the feature for every search window
    while step < steps:
        margin += step_size

        if side == "top":
            frame = canny[0:margin, :]
        elif side == "bottom":
            frame = canny[height - margin:height, :]
        elif side == "left":
            frame = canny[:, 0:margin]
        elif side == "right":
            frame = canny[:, width - margin:width]

        results.append((frame > 0).sum())

        step += 1

    # calculate the gradients of the results array with the offset
    gradients = np.gradient(results)[gradient_offset:]

    step_count = find_following_max(gradients)
    step_count = step_count + gradient_offset

    return step_count * step_size


def calc_step_size(img, steps, max_frame_size, axis=0):
    """
    Calculates the step size with the image size, the steps and the maximal size of the search window.

    :param img: ndarray - The image to get the width or height.
    :param steps: int - The number of steps.
    :param max_frame_size: float - The maximal window size.
    :param axis: [0, 1] The axis of the image (x or y) to calculate the step size for.
    :return: The step size.
    :rtype int
    """
    height, width = img.shape

    if axis == 1:
        step_size = np.ceil((width * max_frame_size) / steps)
    else:
        step_size = np.ceil((height * max_frame_size) / steps)

    return int(step_size)


def find_following_max(gradients):
    """
    Finds the next maximum after the global minimum. If the maximum is to far away from the minimum,
    the minimum is returned.

    :param gradients: array - The gradients.
    :return: int - The steps needed to find the following maximum.
    :rtype int
    """
    argmin = np.argmin(gradients)

    step_count = argmin

    length = len(gradients) - 1

    while step_count < length and gradients[step_count] <= gradients[step_count + 1]:
        if gradients[step_count] == gradients[step_count + 1]:
            argmin = step_count + 1
        step_count += 1

    if step_count - argmin > np.ceil(len(gradients) * 0.2):
        step_count = argmin

    return step_count
