import cv2 as cv
import numpy as np

# Display an OpenCV Image
def show(frame, title='name', wait=False):
    cv.imshow(title, frame)
    if wait:
        cv.waitKey(0)
    else:
        cv.waitKey(1)

# Compute X, Y locations of equidistant points along a line
def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts + 1),
               np.linspace(p1[1], p2[1], parts + 1))