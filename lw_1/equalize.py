import numpy as np
import cv2 as cv

def equalize(img_file, show_hists):

    img = cv.imread(img_file)

    img_bgr = cv.split(img)
    img_final = []

    for i, layer in enumerate(img_bgr):
        img_new = np.clip((cv.equalizeHist(layer)), 0, 255)
        img_final.append(img_new)

    img_new = cv.merge(img_final)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_9.png")

    return img_new