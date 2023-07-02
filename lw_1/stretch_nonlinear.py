import numpy as np
import cv2 as cv

def stretch_nonlinear(img_file, alpha, show_hists):

    img = cv.imread(img_file)

    # Convert to floating point
    img_new = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_bgr = cv.split(img_new)
    img_new_bgr = []

    # Process layers separately
    for layer in img_bgr:

        img_min = layer.min()
        img_max = layer.max()
        img_new = np.clip((((layer - img_min) / (img_max - img_min)) ** alpha), 0, 1)
        img_new_bgr.append(img_new)

    # Merge back
    img_new = cv.merge(img_new_bgr)

    # Convert back to uint if needed
    if img.dtype == np.uint8:
        img_new = (255 * img_new).clip(0, 255).astype(np.uint8)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_3.png")

    return img_new