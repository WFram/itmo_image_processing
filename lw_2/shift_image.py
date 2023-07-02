import cv2 as cv
import numpy as np


def shift_image(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # No need to take layers
    rows, cols = img.shape[:2]
    T = np.float32([[1, 0, 200], [0, 1, 400]])
    img_shifted = cv.warpAffine(img, T, (cols, rows))
