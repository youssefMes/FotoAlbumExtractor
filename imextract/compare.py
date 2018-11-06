import skimage.measure as skimage
import numpy as np
import cv2


def compare_size(truth, img):
    """
    Returns the computed difference in size between the image and the ground truth. 
    The difference is represented as percentage which is positive if the image is greater than the ground truth
    and negative if the image is smaller.

    :param img: this is the image, which will be compared to the second parameter.
    :param truth: this is the image with the ground-truth for the comparison.
    :type img: Mat - n-dimensional dense array to represent an image.
    :type truth: Mat - n-dimensional dense array to represent an image.
    :returns: The difference as Percentage.  
    :rtype: float  
    """
    # compute image size
    img_width, img_height, _ = img.shape
    truth_width, truth_height, _ = truth.shape

    img_size = img_width * img_height
    truth_size = truth_width * truth_height

    # compute difference in size
    diff = img_size - truth_size

    # computer percentage in difference
    return (diff / truth_size) * 100

    
def compare_feature(truth, img, max_features, threshold):
    """
    Returns the computed difference between features of the image and the ground truth.
    This method uses ORB to detect features.

    :param img: this is the image, which will be compared to the second parameter.
    :param truth: this is the image with the groundtruth for the comparison.
    :param max_features: this is the maximum count of used features to compare the two images.
    :param threshold: this is the maximum distance between two fitting features. 
    :type img: ndarray - n-dimansional dense array to represent an image.
    :type truth: ndarray - n-dimansional dense array to represent an image. 
    :type max_features: int
    :type threshold: float
    :returns: The difference Percentage.
    :rtype: float
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    truth = cv2.cvtColor(truth, cv2.COLOR_RGB2GRAY)

    # create ORB object which finds given number of features 
    orb = cv2.ORB.create(max_features)

    # find and compute the keypoints and the corresonding descriptors with ORB
    img_kp, img_des = orb.detectAndCompute(img, None)
    truth_kp, truth_des = orb.detectAndCompute(truth, None)

    # match features with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(img_des, truth_des)

    # compute image center
    img_height, img_width = img.shape
    truth_height, truth_width = truth.shape

    img_center = ( img_width / 2, img_height / 2 )
    truth_center = ( truth_width / 2, truth_height / 2 )

    diffs = []

    # compare matched features to compute differece
    for i, match in enumerate(matches):

        if match.distance < threshold:
            # uses image center to compare the feature position
            img_x = abs( img_center[0] - img_kp[match.queryIdx].pt[0] ) / img_width
            img_y = abs( img_center[1] - img_kp[match.queryIdx].pt[1] ) / img_height

            truth_x = abs( truth_center[0] - truth_kp[match.trainIdx].pt[0] ) / truth_width
            truth_y = abs( truth_center[1] - truth_kp[match.trainIdx].pt[1] ) / truth_height

            diff_x = ( abs(truth_x - img_x) / truth_x ) if truth_x > img_x else ( abs(truth_x - img_x) / img_x )
            diff_y = ( abs(truth_y - img_y) / truth_y ) if truth_y > img_y else ( abs(truth_y - img_y) / img_y )

            diff = (diff_x + diff_y) / 2

            diffs.append(diff)

    if not diffs:
        return 100.0

    return (np.average(diffs)) * 100


def ssim(truth, img):
    """
    Shrinks the images to the size of 64x64 using linear interpolation. 
    Then calculates structural similarity index metric (SSIM)

    Parameters
    ----------
    truth:  ndarray - Groundtruth image  
    img:    ndarray - Image for comparison
    
    Returns
    -------
    ssim:   float - SSIM value between [-1,1]
    """

    size = (64,64)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    truth = cv2.cvtColor(truth, cv2.COLOR_RGB2GRAY)

    img_mini = cv2.resize(img, size)
    truth_mini = cv2.resize(truth, size)

    return skimage.compare_ssim(truth_mini, img_mini)
    