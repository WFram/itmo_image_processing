import cv2 as cv
import numpy as np


def filter_laplace(img_name):

    img_src = cv.imread(img_name)
    img_src = img_src.astype(np.float32) / 255

    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    img_dst = cv.filter2D(img_src, -1, kernel)

    img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    return img_dst
