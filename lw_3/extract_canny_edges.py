import cv2 as cv
import numpy as np


def extract_canny_edges(img_name):

    img_src = cv.imread(img_name)
    img_dst = cv.Canny(img_src, 100, 200)

    return img_dst
