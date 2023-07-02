import cv2 as cv
import numpy as np


def bevel_image(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    # Bevel value
    s = 0.6
    T = np.float32([[1, s, 0],
                    [0, 1, 0]])
    img_bev = cv.warpAffine(img, T, (cols, rows))
