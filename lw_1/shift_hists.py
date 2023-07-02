import numpy as np
import cv2 as cv

def shift_hists(img_file, shift_val, show_hists):

    img = cv.imread(img_file)
    img_new = np.copy(img)

    # Create lookup table
    LUT = []
    for i in range(256):
        LUT.append(255) if i + shift_val > 255 else LUT.append(i + shift_val)

    # Apply lookup table to an input image
    LUT = np.array(LUT, dtype=np.uint8)
    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]
    img_b = LUT[img_b]
    img_g = LUT[img_g]
    img_r = LUT[img_r]
    img_new[:, :, 0] = img_b
    img_new[:, :, 1] = img_g
    img_new[:, :, 2] = img_r

    if show_hists:
        calc_hists_plt(img, img_new, "pic_2.png")
