import argparse
import cv2
import os
import configparser
import sys

import numpy as np

import imextract.compare as cp
import imextract.backgroundremover as br
import imextract.rectextract as re
import imextract.facedetection as fd


def main(args, config):
    """
    TODO: Comment
    :param args:
    :param config:
    :return:
    """

    path_frontal_classifier = config.get('FaceDetection', 'CascadePathFrontal')
    path_profile_classifier = config.get('FaceDetection', 'CascadePathProfile')
    if not path_frontal_classifier or not path_profile_classifier:
        sys.exit("Error in the config file!")

    frontal_classifier = cv2.CascadeClassifier(path_frontal_classifier)
    profile_classifier = cv2.CascadeClassifier(path_profile_classifier)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)

    if img is None:
        sys.exit("Error no image found!")

    if args.compare:
        compare_images(img, args.compare)
        return

    full_name = os.path.basename(args.image)
    name = os.path.splitext(full_name)[0]

    cropped_images = get_background_extracted_images(img, config)

    for i, image in enumerate(cropped_images):

        img = get_frame_extracted_image(image, config)

        if args.face:
            get_detected_faces(img, frontal_classifier, profile_classifier, config)

        cv2.imwrite(args.result + '/' + name + '_' + str(i) + '.png', img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("image",
                        help="path to the image of the photo album page",
                        )

    parser.add_argument("--result",
                        help="path to the directory to store the results",
                        type=str,
                        default="./output"
                        )

    parser.add_argument("--compare",
                        help="path to the image to compare the given image to",
                        type=str
                        )

    parser.add_argument("--config",
                        help="path to the config file",
                        type=str,
                        default="./config"
                        )

    parser.add_argument("--face",
                        help="enables face detection",
                        action="store_true"
                        )

    arguments = parser.parse_args()

    configuration = configparser.ConfigParser()
    configuration.read(arguments.config)

    main(arguments, configuration)


def compare_images(img, path):
    """
    TODO: Comment

    :param img:
    :param path:
    :return:
    """
    truth = cv2.imread(path, cv2.IMREAD_COLOR)
    print("Difference percentage of the size: {} %".format(cp.compare_size(truth, img)))
    print("Difference percentage of the features: {} %".format(cp.compare_feature(truth, img, 20, 31.0)))
    print("SSIM: {} %".format(cp.ssim(truth, img)))


def get_frame_extracted_image(img, config):
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
    if not max_window_size or not steps or not offset:
        sys.exit("Error in the config file!")

    img = re.remove_boarder(img, steps, max_window_size, offset)

    return img


def get_background_extracted_images(img, config):
    """
    Approximately cuts out all the photos in the input image.

    :param img: the scanned site of an album
    :returns: array of cut out images
    :rtype: array
    """

    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)

    background_color = br.get_primary_background_color(img)

    spot_size = config.get('BackgroundRemover', 'SpotSize')
    if not spot_size:
        sys.exit("Error in the config file!")

    background_location = br.get_background_spot(img, background_color, spot_size)

    binary_threshold = config.get('BackgroundRemover', 'BinaryThreshold')
    if not binary_threshold:
        sys.exit("Error in the config file!")

    binary_img = br.generate_binary_background_image(img, background_color, binary_threshold)
    binary_background_img = br.separate_background(binary_img, background_location)

    min_area = config.get('BackgroundRemover', 'MinImageSize')
    max_dimension_relation = config.get('BackgroundRemover', 'MaxRelationImageDimensions')
    image_padding = config.get('BackgroundRemover', 'ImagePadding')
    if not min_area or not max_dimension_relation or not image_padding:
        sys.exit("Error in the config file!")

    cropped_images = br.crop_image_rectangles(img, binary_background_img)

    feature_threshold = config.get()
    if not feature_threshold:
        sys.exit("Error in the config file!")

    valid_cropped_images = br.validate_cropped_images(cropped_images, feature_threshold)

    return valid_cropped_images


def get_detected_faces(img, frontal_classifier, profile_classifier, config):
    """
    TODO: Comment

    :param img:
    :param frontal_classifier:
    :param profile_classifier:
    :param config:
    :return:
    """

    scale = config.getfloat('FaceDetection', 'ScaleFactor')
    neighbors = config.getint('FaceDetection', 'Neighbors')
    if not scale or not neighbors:
        sys.exit("Error in the config file!")

    faces_list = fd.detect_faces(img, frontal_classifier, profile_classifier, scale, neighbors)

    return fd.mark_faces(img, faces_list)