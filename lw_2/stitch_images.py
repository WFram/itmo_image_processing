import cv2 as cv
import numpy as np


def stitch_images(img_left_name, img_right_name):

    img_left = cv.imread(img_left_name)
    img_right = cv.imread(img_right_name)

    # Create a template for calculating correlation (10 columns from the left image)
    templ_size = 10
    templ = img_left[-templ_size:, :, :]
    res = cv.matchTemplate(img_right, templ, cv.TM_CCOEFF)

    # Choose the stitching point as the point with max correlation value
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # Create stitched image
    img_dst = np.zeros((img_left.shape[0],
                        img_left.shape[1] + img_right.shape[1] - max_loc[1] - templ_size,
                        img_left.shape[2]),
                       dtype=np.uint8)

    img_dst[:, 0:img_left.shape[1], :] = img_left
    img_dst[:, img_left.shape[1]:, :] = img_right[:, max_loc[1] + templ_size:, :]
