import cv2 as cv


def filter_gaussian(img_name, kernel=7, sigma=0.0):
    img_src = cv.imread(img_name)

    img_dst = cv.GaussianBlur(img_src, (kernel, kernel), sigma)

    return img_dst
