import argparse
import cv2
import os
import configparser

import imextract.compare as cp
import imextract.backgroundremover as br
import imextract.rectextract as re
import imextract.facedetection as fd


def main(args, config):

    frontal_classifier = cv2.CascadeClassifier(config.get('FaceDetection', 'CascadePathFrontal'))
    profile_classifier = cv2.CascadeClassifier(config.get('FaceDetection', 'CascadePathProfile'))

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)

    if args.compare:
        truth = cv2.imread(args.compare, cv2.IMREAD_COLOR)
        print("Difference percentage of the size: {} %".format(cp.compare_size(truth, img)))
        print("Difference percentage of the features: {} %".format(cp.compare_feature(truth, img, 20, 31.0)))
        print("SSIM: {} %".format(cp.ssim(truth, img)))
    else:
        full_name = os.path.basename(args.image)
        name = os.path.splitext(full_name)[0]

        cropped_images = br.process_image(img, config)

        for i in range(len(cropped_images)):
            img = re.process_image(cropped_images[i], config)

            if args.face:
                faces_list = fd.detect(img,
                                       frontal_classifier,
                                       profile_classifier,
                                       config.getfloat('FaceDetection', 'ScaleFactor'),
                                       config.getint('FaceDetection', 'Neighbors')
                                       )

                img = fd.mark_faces(img, faces_list)

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
