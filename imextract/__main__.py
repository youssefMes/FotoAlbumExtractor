import sys
import argparse
import cv2
import numpy as np
import compare as cp
import backgroundremover as br


def main(args=None):

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)

    if args.compare:
        truth = cv2.imread(args.compare, cv2.IMREAD_COLOR)
        print("Difference percentage of the size: {}".format(cp.compare_size(truth, img)))
        print("Difference percentage of the features: {}".format(cp.compare_feature(thruth, img, 20, 0.5)))
    else:
        if args.path:
            br.processImage(img, args.path)
        else:
            br.processImage(img)


def deleteFolderContent(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        return

    for the_file in os.listdir(folderpath):
        file_path = os.path.join(folderpath, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to the image of the photo album page")
    parser.add_argument("-p", "--path", help="path to the directory to store the results", type=str)
    parser.add_argument("-c", "--compare", help="path to the image to compare the given image to", type=str)
    args = parser.parse_args()

    main(args)