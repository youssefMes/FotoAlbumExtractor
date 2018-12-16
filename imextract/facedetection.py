import cv2


def detect_faces(img, frontal_classifier, profile_classifier, scale,
                 neighbors):
    """
    This detects faces in the given image. It detects faces in profile and frontal view.
    If the two found faces are to near to each other the profile detection is removed to avoid a
    double detection of the same face.

    :param img: ndarray - The image to detect faces.
    :param frontal_classifier: The Casscade classifier for frontal faces.
    :param profile_classifier: The Casscade classifier for profile faces.
    :param scale: specifies how much the image size is reduced at each image scal
    :param neighbors: specifies how many neighbors each candidate rectangle should have to retain it.
    :return:
    """

    # convert the image to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # frontal faces
    frontal = frontal_classifier.detectMultiScale(img_gray, scale, neighbors)
    # profile faces
    profile = profile_classifier.detectMultiScale(img_gray, scale, neighbors)

    frontal_list = list(frontal)
    profile_list = list(profile)

    #checks if faces are too close to each other
    if profile_list:
        for x, y, w, h in frontal_list:
            for i, profile in enumerate(profile_list):
                x_p = profile[0]
                y_p = profile[1]
                w_p = profile[2]
                h_p = profile[3]
                center_p = ((x_p + w_p / 2), (y_p + h_p / 2))
                if center_p[0] > x and center_p[0] < x + w and center_p[1] > y and center_p[1] < y + h:
                    del profile_list[i]

    frontal_list.extend(profile_list)

    return frontal_list


def mark_faces(img, faces_list):
    """
    The found faces in the list of faces are marked with a green rectangle in the image.

    :param img: ndarray - The image to mark the faces.
    :param faces_list: array - The detected faces in the image.
    :return: The image with the marked faces.
    :rtype ndarray
    """

    if faces_list:
        for (x, y, w, h) in faces_list:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img