import cv2


def detect(img, frontal_classifier, profile_classifier, scale=1.2, neighbors=5):

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

    return frontal_list


def save_faces(img, faces_list, output):
    if faces_list:
        i = 0
        for (x, y, w, h) in faces_list:
            sub_face = img[y:y + h, x:x + w]
            cv2.imwrite(output + "test" + "_" + str(i) + ".png", sub_face)
            i += 1


def mark_faces(img, faces_list):
    if faces_list:
        for (x, y, w, h) in faces_list:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img
