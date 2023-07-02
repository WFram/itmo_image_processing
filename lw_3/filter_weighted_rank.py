import cv2 as cv
import numpy as np

def filter_weighted_rank(img_name, kernel_size, rank=4, use_median=False, use_weighted=False):
    img = cv.imread(img_name)

    # Filter parameters
    kernel = np.ones(kernel_size, dtype=np.float32)

    # Recalculate the rank if we use median filtering
    if use_median:
        N = kernel_size[0] * kernel_size[1]
        rank = int(0.5 * (N - 1))

    # Weight the kernel if needed
    if use_weighted:
        if kernel_size[0] == 3:
            kernel = np.array([[0.8, 0.9, 0.8],
                               [0.9, 1.0, 0.9],
                               [0.8, 0.9, 0.8]], dtype=np.float32)
        elif kernel_size[0] == 5:
            kernel = np.array([[0.6, 0.7, 0.8, 0.7, 0.6],
                               [0.7, 0.8, 0.9, 0.8, 0.7],
                               [0.8, 0.9, 1.0, 0.9, 0.8],
                               [0.7, 0.8, 0.9, 0.8, 0.7],
                               [0.6, 0.7, 0.8, 0.7, 0.6]], dtype=np.float32)
    rows, cols = img.shape[:2]

    # Convert to float and make image with boarders
    img_copy = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_copy = cv.copyMakeBorder(img_copy,
                                 int((kernel_size[0] - 1) / 2),
                                 int(kernel_size[0] / 2),
                                 int((kernel_size[1] - 1) / 2),
                                 int(kernel_size[1] / 2),
                                 cv.BORDER_REPLICATE)

    # Fill arrays for each kernel item
    img_layers = np.zeros(img.shape + (kernel_size[0] * kernel_size[1],),
                          dtype=np.float32)
    if img.ndim == 2:
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                img_layers[:, :, i * kernel_size[1] + j] = kernel[i, j] * img_copy[i:i + rows, j:j + cols]
    else:
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                img_layers[:, :, :, i * kernel_size[1] + j] = kernel[i, j] * img_copy[i:i + rows, j:j + cols, :]

    # Sort arrays
    img_layers.sort()

    # Choose layer with rank
    img_dst = img_layers[:, :, rank] if img.ndim == 2 else img_layers[:, :, :, rank]

    # Convert back to uint as necessary
    if img.dtype == np.uint8:
        img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    return img_dst
