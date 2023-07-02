import cv2 as cv
import numpy as np


def reflect_image(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    T = np.float32([[-1, 0, cols - 1], [0, 1, 0]])
    img_reflected = cv.warpAffine(img, T, (cols, rows))
