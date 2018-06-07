import numpy as np
import cv2


def compare_size(truth, img):
    '''
    Returns the computed difference in size between the image and the ground truth. 
    The difference is represented as percentage which is positive if the image is greater than the ground truth
    and negative if the image is smaller.

    :param image: this is the image, which will be compared to the second parameter. 
    :param truth: this is the image with the groundtruth for the comparison. 
    :type image: Mat - n-dimansional dense array to represent an image. 
    :type truth: Mat - n-dimansional dense array to represent an image. 
    :returns: The difference as Percentage.
    :rtype: float 
    '''   
    # compute image size
    img_width, img_height = img.shape
    truth_width, truth_height = truth.shape

    img_size = img_width * img_height
    truth_size = truth_width * truth_height

    # compute difference in size
    diff = img_size - truth_size

    # computer percentage in difference
    return (diff / truth_size) * 100

    
def compare_feature(truth, img, features, threshold):
    '''
    Returns the computed difference between features of the image and the ground truth.
    This method uses ORB to detect features.
    :param image: this is the image, which will be compared to the second parameter. 
    :param truth: this is the image with the groundtruth for the comparison.
    :param features: this is the count of used features to compare the two images.
    :param threshold: this is the maximum distance between two fitting features. 
    :type image: Mat - n-dimansional dense array to represent an image. 
    :type truth: Mat - n-dimansional dense array to represent an image. 
    :type features: int
    :type threshold: float
    :returns: The difference Percentage.
    :rtype: float
    '''
    # create ORB object which finds given number of features 
    orb = cv2.ORB.create(features)

    # find and compute the keypoints and the corresonding descriptors with ORB
    img_kp, img_des = orb.detectAndCompute(img, None)
    truth_kp, truth_des = orb.detectAndCompute(truth, None)

    # match features with Hamming distance
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(img_des, truth_des)

    # compute image center
    img_height, img_width = img.shape
    truth_height, truth_width = truth.shape

    img_center = ( img_width / 2, img_height / 2 )
    truth_center = ( truth_width / 2, truth_height / 2 )

    diffs = np.empty( len(matches) )

    # compare matched features to compute differece
    for i, match in enumerate(matches):
        #print("Distance: {}".format(match.distance) )
        #print("QueriIdx: {}".format(match.queryIdx) )
        #print("TrainIdx: {}".format(match.trainIdx) )
        #print("===============================")

        if match.distance > threshold:
            # if the distance between the feature is too big
            diffs[i] = 1.0
        else:
            # uses image center to compare the feature position
            img_x = abs( img_center[0] - img_kp[match.queryIdx].pt[0] )
            img_y = abs( img_center[1] - img_kp[match.queryIdx].pt[1] )

            truth_x = abs( truth_center[0] - truth_kp[match.trainIdx].pt[0] )
            truth_y = abs( truth_center[1] - truth_kp[match.trainIdx].pt[1] )

            diffs[i]  = ( abs(truth_x - img_x) / truth_x ) + ( abs(truth_y - img_y) / truth_y )

    return np.average(diffs) * 100
