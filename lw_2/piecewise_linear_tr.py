import cv2 as cv
import numpy as np


def piecewise_linear_tr(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    stretch = 3
    T = np.float32([[stretch, 0, 0],
                    [0, 1, 0]])
    img_tr = img.copy()
    # Stretch only right part of an image
    img_tr[:, int(cols / 2):, :] = cv.warpAffine(img_tr[:, int(cols / 2):, :],
                                                 T,
                                                 (cols - int(cols / 2), rows))
