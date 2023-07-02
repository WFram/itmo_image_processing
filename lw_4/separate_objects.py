import cv2 as cv
import numpy as np


def separate_objects(img_name):

    img_src = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    ret, img_dst = cv.threshold(img_src, 160, 255, cv.THRESH_BINARY_INV)

    kernel_size = 7
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                      (kernel_size, kernel_size))

    # Erosion
    bw_2 = cv.morphologyEx(img_dst,
                           cv.MORPH_ERODE,
                           kernel,
                           iterations=30,
                           borderType=cv.BORDER_CONSTANT,
                           borderValue=(0))
    # Dilation
    t = np.zeros_like(img_dst)
    while cv.countNonZero(bw_2) < bw_2.size:

        d = cv.dilate(bw_2,
                      kernel,
                      borderType=cv.BORDER_CONSTANT,
                      borderValue=(0))

        c = cv.morphologyEx(d,
                            cv.MORPH_CLOSE,
                            kernel,
                            borderType=cv.BORDER_CONSTANT,
                            borderValue=(0))
        s = c - d
        t = cv.bitwise_or(s, t)
        bw_2 = d

    # Closing for borders
    t = cv.morphologyEx(t,
                        cv.MORPH_CLOSE,
                        kernel,
                        iterations=31,
                        borderType=cv.BORDER_CONSTANT,
                        borderValue=(0))

    # Remove borders from an image
    t_inv = np.logical_not(t)

    t_inv = t_inv.astype(np.float32)
    img_dst = img_dst.astype(np.float32)
    img_dst = cv.bitwise_and(t_inv, img_dst)
    img_dst = np.logical_not(img_dst)

    return img_dst
