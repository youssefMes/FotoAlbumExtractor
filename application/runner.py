from mtcnn.mtcnn import MTCNN

import argparse
import configparser
import cv2
import os

import extractor.extract as extractor
import detector.detect as detector
import enhancer.enhance as enhancer
import enhancer.mtcn_face as mtcn_face



def main(args, config):
    """
    Main function to run the program with the given arguments.

    :param args: dictionary - The Arguments given by the start of the programm.
    :param config: dictionary - The configuration of the config file.
    """
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)

    if img is None:
        sys.exit("Error no image found!")

    if args.compare:
        compare_images(img, args.compare)
    else:
        full_name = os.path.basename(args.image)
        name = os.path.splitext(full_name)[0]

        cropped_images = extractor.get_background_extracted_images(img, config)

        for i, image in enumerate(cropped_images):

            img = extractor.get_frame_extracted_image(image, config)
            if args.enhance:
                print('enhance true')
                img = enhancer.enhance_image(img)
            if args.face:

                path_frontal_classifier = config.get('FaceDetection', 'CascadePathFrontal')
                path_profile_classifier = config.get('FaceDetection', 'CascadePathProfile')

                if not path_frontal_classifier or not path_profile_classifier:
                    sys.exit("Error in the config file!")

                frontal_classifier = cv2.CascadeClassifier(path_frontal_classifier)
                profile_classifier = cv2.CascadeClassifier(path_profile_classifier)

                #detector.get_detected_faces(img, frontal_classifier, profile_classifier, config, args.result, name + '_' + str(i))
                mtcnnDetector = MTCNN()
                # detect faces in the image
                faces = mtcnnDetector.detect_faces(img)
                # display faces on the original image
                pyplot = mtcn_face.draw_image_with_boxes(img, faces, name + '_' + str(i))
            if not os.path.exists(args.result):
                os.makedirs(args.result)

            pyplot.savefig(args.result + '/' + name + '_' + str(i) + '.png', bbox_inches = 'tight', facecolor='white')
            # show the plot
            pyplot.show()
            #cv2.imwrite(args.result + '/' + name + '_' + str(i) + '.png', img)


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
    parser.add_argument("--enhance",
                        help="Enhance images before detection",
                        action="store_true"
                        )

    arguments = parser.parse_args()

    configuration = configparser.ConfigParser()
    configuration.read(arguments.config)

    main(arguments, configuration)
