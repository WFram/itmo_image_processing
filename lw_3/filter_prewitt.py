import cv2 as cv
import numpy as np


def filter_prewitt(img_name):

    img_src = cv.imread(img_name)
    img_src = img_src.astype(np.float32) / 255

    g_x = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])
    g_y = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])
    i_x = cv.filter2D(img_src, -1, g_x)
    i_y = cv.filter2D(img_src, -1, g_y)
    img_dst = cv.magnitude(i_x, i_y)

    img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    return img_dst
