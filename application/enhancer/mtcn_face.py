import os
import cv2
# face detection with mtcnn
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle


# draw an image with detected objects
def draw_image_with_boxes(img, result_list, name):
    # get the context for drawing boxes
    pyplot.imshow(img)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box

    faces_path = './output/faces/'
    if not os.path.exists(faces_path):
        os.makedirs(faces_path)

    j = 0
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']

        #save img
        face = img[y:y+height, x:x+width]
        cv2.imwrite(faces_path + name + '_' + str(j) + ".png", face)
        j += 1

        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)

    # show the plot
    #pyplot.show()
    return pyplot


