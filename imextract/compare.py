import numpy as np
import cv2


class ImageCompare:
    '''
    This is an object to compare two images.

    :param image: this is the image, which will be compared to the second parameter.
    :param groundtruth: this is the image with the groundtruth for the comparison.
    :type image: Mat - n-dimansional dense array to represent an image.
    :type groundtruth: Mat - n-dimansional dense array to represent an image.
    '''
    def __init__(self, image, groundtruth, maxfeatures):
        self.__img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.__truth = cv2.cvtColor(groundtruth, cv2.COLOR_BGR2GRAY)
        self.__max_features = maxfeatures
        self.__diff_size = None
        self.__diff_feature = None


    def compare_size(self):
        '''
        Returns the computed difference in size between the image and the ground truth. 
        The difference is represented as percentage which is positive if the image is greater than the ground truth
        and negative if the image is smaller.

        :returns: The difference as Percentage.
        :rtype: float 
        '''
        if self.__diff_size:
            return self.__diff_size
        
        # compute image size
        img_width, img_height = self.__img.shape
        truth_width, truth_height = self.__truth.shape

        img_size = img_width * img_height
        truth_size = truth_width * truth_height

        # compute difference in size
        diff = img_size - truth_size

        # computer percentage in difference
        self.__diff_size = (diff / truth_size) * 100

        return self.__diff_size

    
    def compare_feature(self):
        '''
        Returns the computed difference between features of the image and the ground truth.
        This method uses ORB to detect features.
        '''
        if self.__diff_feature:
            return self.__diff_feature

        # create ORB object which finds given number of features 
        orb = cv2.ORB.create(self.__max_features)

        # find and compute the keypoints and the corresonding descriptors with ORB
        img_kp, img_des = orb.detectAndCompute(self.__img, None)
        truth_kp, truth_des = orb.detectAndCompute(self.__truth, None)

        # match features with Hamming distance
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(img_des, truth_des)

        # compute image center
        img_height, img_width = self.__img.shape
        truth_height, truth_width = self.__truth.shape

        img_center = ( img_width / 2, img_height / 2 )
        truth_center = ( truth_width / 2, truth_height / 2 )

        diffs = np.empty( len(matches) )

        # compare matched features to compute differece
        for i, match in enumerate(matches):
            print("Distance: {}".format(match.distance) )
            print("QueriIdx: {}".format(match.queryIdx) )
            print("TrainIdx: {}".format(match.trainIdx) )
            print("===============================")

            if match.distance > 0.5:
                # if the distance between the feature is too big
                diffs[i] = 1.0
            else:
                # uses image center to compare the feature position
                img_x = abs( img_center[0] - img_kp[match.queryIdx].pt[0] )
                img_y = abs( img_center[1] - img_kp[match.queryIdx].pt[1] )

                truth_x = abs( truth_center[0] - truth_kp[match.trainIdx].pt[0] )
                truth_y = abs( truth_center[1] - truth_kp[match.trainIdx].pt[1] )

                diffs[i]  = ( abs(truth_x - img_x) / truth_x ) + ( abs(truth_y - img_y) / truth_y )

        self.__diff_feature = np.average(diffs) * 100

        return self.__diff_feature