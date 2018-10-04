import cv2


def detect(img, frontal_classifier, profile_classifier, config):

    scale = config.getfloat('FaceDetection', 'ScaleFactor')
    neighbors = config.getint('FaceDetection', 'Neighbors')

    # convert the image to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # frontal faces
    frontal = frontal_classifier.detectMultiScale(img_gray, scale, neighbors)
    # profile faces
    profile = profile_classifier.detectMultiScale(img_gray, scale, neighbors)

    frontal_list = list(frontal)
    profile_list = list(profile)

    for x, y, w, h in frontal_list:
        for i, profile in enumerate(profile_list):
            x_p = profile[0]
            y_p = profile[1]
            w_p = profile[2]
            h_p = profile[3]
            # checks if faces are to close to each other
            if float(abs(y_p - y)) < h_p / 3 and float(abs(x_p - x)) < w_p / 3:
                del profile_list[i]

    frontal_list.extend(profile_list)

    img = mark_faces(img, frontal_list)

    return img


def mark_faces(img, faces_list):
    if faces_list:
        for (x, y, w, h) in faces_list:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img
