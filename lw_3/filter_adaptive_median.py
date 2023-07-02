import cv2 as cv
import numpy as np


def adaptive_filter_iterate(img, i, j, kernel_size, s_max):

    coords = []

    # Find image pixels in the sliding window
    for m in range(-int(kernel_size / 2), int(kernel_size / 2) + 1):
        for n in range(-int(kernel_size / 2), int(kernel_size / 2) + 1):
            coords.append(img[i + m, j + n, 0])

    coords.sort()
    z_min = coords[0]
    z_max = coords[-1]
    z_max = coords[int(kernel_size * kernel_size - 1)]
    z_med = coords[int(kernel_size * kernel_size / 2)]
    z_xy = img[i, j, 0]

    ret_val = z_med
    if z_min < z_med < z_max:
        ret_val = z_xy if z_min < z_xy < z_max else z_med
    else:
        kernel_size += 2
        ret_val = adaptive_filter_iterate(img, i, j, kernel_size, s_max) if kernel_size <= s_max else z_med

    return ret_val


def filter_adaptive_median(img_name, min_kernel_size=3, max_kernel_size=5):

    img_src = cv.imread(img_name)

    s_min = min_kernel_size
    s_max = max_kernel_size

    # Convert to float and make an image with borders
    img_copy = img_src.astype(np.float32) / 255 if img_src.dtype == np.uint8 else img_src
    img_copy = cv.copyMakeBorder(img_copy,
                                 int((s_min / 2) - 1),
                                 int(s_max),
                                 int((s_min / 2) - 1),
                                 int(s_max),
                                 cv.BORDER_REPLICATE)

    rows, cols = img_copy.shape[:2]

    # Iterate over the whole image
    for i in range(int(s_max / 2), rows - int(s_max / 2)):
        for j in range(int(s_max / 2), cols - int(s_max / 2)):
            img_copy[i, j] = adaptive_filter_iterate(img_copy, i, j, s_min, s_max)

    img_dst = np.copy(img_copy)

    if img_src.dtype == np.uint8:
        img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    return img_dst
