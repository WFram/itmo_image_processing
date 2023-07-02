import numpy as np
import cv2 as cv

def law_2_3(img_file, show_hists):

    img = cv.imread(img_file)

    img_new = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_bgr = cv.split(img)
    img_new_bgr = cv.split(img_new)
    img_final = []

    hist = calc_hist_cv(img_bgr)
    for i, layer in enumerate(img_new_bgr):
        cum_hist = (255 * (np.cumsum(hist[i])) / (layer.shape[0] * layer.shape[1])).clip(0, 255).astype(np.uint8)
        layer = (255 * layer).clip(0, 255).astype(np.uint8)
        cum_hist = (cum_hist.astype(np.float32) / 255).clip(0, 1)
        img_new = np.clip((cum_hist[layer] ** (2 / 3)), 0, 1)
        img_final.append(img_new)

    img_new = cv.merge(img_final)

    if img.dtype == np.uint8:
        img_new = (255 * img_new).clip(0, 255).astype(np.uint8)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_7.png")

    return img_new