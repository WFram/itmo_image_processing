import cv2 as cv
import numpy as np


def scale_image(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    s = 2  # Scale
    T = np.float32([[s, 0, 0], [0, s, 0]])
    img_scaled = cv.warpAffine(img, T, (cols, rows))
