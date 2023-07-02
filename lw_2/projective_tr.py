import cv2 as cv
import numpy as np


def projective_tr(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    T = np.float32([[0.9, 0.2, 0.0],
                    [0.35, 0.9, -rows / 4]])
    img_tr = cv.warpAffine(img, T, (cols, rows))
