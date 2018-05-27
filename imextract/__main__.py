import sys
import argparse
import cv2
import numpy as np
import compare


def main(args=None):

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)

    if args.compare:
        truth = cv2.imread(args.compare, cv2.IMREAD_COLOR)
        imcomp = compare.ImageCompare(img, truth, 20)
        print("Difference percentage of the size: {}".format(imcomp.compare_size()))
        print("Difference percentage of the features: {}".format(imcomp.compare_feature()))
    else:
        print("This is the main routine.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to the image of the photo album page")
    parser.add_argument("-p", "--path", help="path to the directory to store the results", type=str)
    parser.add_argument("-c", "--compare", help="path to the image to compare the given image to", type=str)
    args = parser.parse_args()

    main(args)