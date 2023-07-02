import cv2 as cv
import numpy as np


def filter_wiener(img_name, kernel_size=(9, 9)):

    img = cv.imread(img_name)

    # Filter parameters
    kernel = np.ones(kernel_size)
    rows, cols = img.shape[:2]

    # Convert to float and make image with boarders
    img_copy = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_copy = cv.copyMakeBorder(img_copy,
                                 int((kernel_size[0] - 1) / 2),
                                 int(kernel_size[0] / 2),
                                 int((kernel_size[1] - 1) / 2),
                                 int(kernel_size[1] / 2),
                                 cv.BORDER_REPLICATE)

    # Split into layers
    layers = cv.split(img_copy)
    layers_new = []
    k_power = np.power(kernel, 2)
    for layer in layers:

        # Calculate temporary matrices
        layer_power = np.power(layer, 2)
        m = np.zeros(img.shape[:2], np.float32)
        q = np.zeros(img.shape[:2], np.float32)

        # Calculate variance values
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                m = m + kernel[i, j] * layer[i:i + rows, j:j + cols]
                q = q + k_power[i, j] * layer_power[i:i + rows, j:j + cols]

        m /= np.sum(kernel)
        q /= np.sum(kernel)
        q = q - m * m

        # Calculate noise as an average variance
        v = np.sum(q) / img.size

        # Perform filtering
        layer_new = layer[(kernel_size[0] - 1) // 2:(kernel_size[0] - 1) // 2 + rows,
                    (kernel_size[1] - 1) // 2:(kernel_size[1] - 1) // 2 + cols]
        layer_new = np.where(q < v, m, (layer_new - m) * (1 - v / q) + m)
        layers_new.append(layer_new)

    # Merge image back
    img_dst = cv.merge(layers_new)

    # Convert back to uint as necessary
    if img.dtype == np.uint8:
        img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    return img_dst
