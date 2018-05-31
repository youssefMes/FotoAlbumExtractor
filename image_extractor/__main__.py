import sys
import argparse


def main(args=None):
    if args.path:
        print("given path: %s" % args.path)
    print("This is the main routine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to the image of the photo album page")
    parser.add_argument("-p", "--path", help="path to the directory to store the results", type=str)
    args = parser.parse_args()

    main(args)