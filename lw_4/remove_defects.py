import cv2 as cv
import numpy as np


def remove_defects(img_name):

    img_src = cv.imread(img_name)
    img_src = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)

    a_kernel = 3
    b_kernel = 6
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (a_kernel, b_kernel))
    img_dst = cv.morphologyEx(img_src,
                              cv.MORPH_OPEN,
                              kernel,
                              iterations=35,
                              borderType=cv.BORDER_CONSTANT,
                              borderValue=(255))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (b_kernel, a_kernel))
    img_dst = cv.morphologyEx(img_dst,
                              cv.MORPH_OPEN,
                              kernel,
                              iterations=55,
                              borderType=cv.BORDER_CONSTANT,
                              borderValue=(255))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (a_kernel, a_kernel))
    img_dst = cv.morphologyEx(img_dst,
                              cv.MORPH_ERODE,
                              kernel,
                              iterations=28,
                              borderType=cv.BORDER_CONSTANT,
                              borderValue=(255))

    return img_dst