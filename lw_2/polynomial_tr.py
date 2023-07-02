import cv2 as cv
import numpy as np


def polynomial_tr(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    T = np.float32([[0, rows / 4], [0.9, 0], [0, 0.9],
                    [0.00001, 0], [0.002, 0], [0.001, 0]])
    img_tr = np.zeros(img.shape, img.dtype)
    x, y = np.meshgrid(np.arange(cols),
                       np.arange(rows))

    # Calculate all new X and Y coordinates
    der_x = T[0, 0] + x * T[1, 0] + y * T[2, 0] + x * x * T[3, 0] + x * y * T[4, 0] + y * y * T[5, 0]
    der_y = T[0, 1] + x * T[1, 1] + y * T[2, 1] + x * x * T[3, 1] + x * y * T[4, 1] + y * y * T[5, 1]
    x_new = np.round(der_x).astype(np.float32)
    y_new = np.round(der_y).astype(np.float32)

    # Calculate a mask of valid indexes
    mask = np.logical_and(np.logical_and(x_new >= 0, x_new < cols),
                          np.logical_and(y_new >= 0, y_new < rows))

    # Apply reindexing
    if img.ndim == 2:
        img_tr[y_new[mask].astype(int), x_new[mask].astype(int)] = img[y[mask], x[mask]]
    else:
        img_tr[y_new[mask].astype(int), x_new[mask].astype(int), :] = img[y[mask], x[mask], :]
