import cv2 as cv
import numpy as np


def filter_counter_harmonic_average(img_name, kernel_size=(3, 3), Q=0):

    img_src = cv.imread(img_name)
    kernel = np.ones(kernel_size)
    kernel /= kernel_size[0] / kernel_size[1]
    rows, cols = img_src.shape[:2]

    # Convert to float and make image with border
    img_copy = img_src.astype(np.float32) / 255 if img_src.dtype == np.uint8 else img_src
    img_copy = cv.copyMakeBorder(img_copy,
                                 int((kernel_size[0] - 1) / 2),
                                 int(kernel_size[0] / 2),
                                 int((kernel_size[1] - 1) / 2),
                                 int(kernel_size[1] / 2),
                                 cv.BORDER_REPLICATE)

    # Split into layers
    layers = cv.split(img_copy)
    layers_new = []
    for layer in layers:

        # Calculate temporary matrices
        m = np.zeros(img_src.shape[:2], np.float32)
        q = np.zeros(img_src.shape[:2], np.float32)
        n = np.power(layer, Q)
        p = np.power(layer, Q + 1)

        # Perform filtering
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                m = m + kernel[i, j] * p[i:i + rows, j:j + cols]
                q = q + kernel[i, j] * n[i:i + rows, j:j + cols]

        layer_new = m / q
        layers_new.append(layer_new)

    # Merge image back
    img_dst = cv.merge(layers_new)

    if img_src.dtype == np.uint8:
        img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)


    return img_dst