import cv2 as cv
import numpy as np


def extract_external_borders(img_name):

    img_src = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    _, img_dst = cv.threshold(img_src, 127, 255, cv.THRESH_BINARY_INV)
    _, img_src = cv.threshold(img_src, 127, 255, cv.THRESH_BINARY_INV)

    kernel_size = 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                      (kernel_size, kernel_size))

    bw_2 = cv.morphologyEx(img_dst,
                           cv.MORPH_DILATE,
                           kernel,
                           iterations=3,
                           borderType=cv.BORDER_CONSTANT,
                           borderValue=(0))

    img_dst = bw_2 - img_src

    return img_dst


def extract_internal_borders(img_name):

    img_src = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    _, img_dst = cv.threshold(img_src, 127, 255, cv.THRESH_BINARY_INV)
    _, img_src = cv.threshold(img_src, 127, 255, cv.THRESH_BINARY_INV)

    kernel_size = 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                      (kernel_size, kernel_size))

    bw_2 = cv.morphologyEx(img_dst,
                           cv.MORPH_ERODE,
                           kernel,
                           iterations=3,
                           borderType=cv.BORDER_CONSTANT,
                           borderValue=(0))

    img_dst = img_src - bw_2

    return img_dst


