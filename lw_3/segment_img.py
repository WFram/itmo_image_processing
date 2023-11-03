import cv2 as cv
import numpy as np


def segment_img(img_name):

    img_src = cv.imread(img_name, cv.IMREAD_COLOR)
    gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
    ret, img_bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    img_fg = cv.distanceTransform(img_bw, cv.DIST_L2, 5)

    ret, img_fg = cv.threshold(img_fg, 0.6 * img_fg.max(), 255, 0)
    img_fg = img_fg.astype(np.uint8)

    ret, markers = cv.connectedComponents(img_fg)

    img_bg = np.zeros_like(img_bw)
    markers_bg = markers.copy()
    markers_bg = cv.watershed(img_src, markers_bg)
    img_bg[markers_bg == -1] = 255

    img_unk = cv.subtract(~img_bg, img_fg)

    markers += 1
    markers[img_unk == 255] = 0

    markers = cv.watershed(img_src, markers)
    img_src[markers == -1] = (0, 0, 255)

    return
