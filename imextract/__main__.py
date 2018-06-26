import sys
import argparse
import cv2
import numpy as np
import compare as cp
import backgroundremover as br
import os


def main(args=None):

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)

    if args.compare:
        truth = cv2.imread(args.compare, cv2.IMREAD_COLOR)
        print("Difference percentage of the size: {} %".format(cp.compare_size(truth, img)))
        print("Difference percentage of the features: {} %".format(cp.compare_feature(truth, img, 20, 31.0)))
        print("SSIM: {} %".format(cp.ssim(truth, img)))
    else:
        full_name = os.path.basename(args.image)
        name = os.path.splitext(full_name)[0]
        if args.path:
            br.processImage(img, name, args.path)
        else:
            br.processImage(img, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to the image of the photo album page")
    parser.add_argument("-p", "--path", help="path to the directory to store the results", type=str)
    parser.add_argument("-c", "--compare", help="path to the image to compare the given image to", type=str)
    args = parser.parse_args()

    main(args)