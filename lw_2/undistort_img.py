import cv2 as cv
import numpy as np


def undistort_img(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]

    # Create mesh grid for X, Y
    x_i, y_i = np.meshgrid(np.arange(cols), np.arange(rows))
    # Shift and normalize grid
    x_mid = cols / 2.0
    y_mid = rows / 2.0
    x_i = x_i - x_mid
    y_i = y_i - y_mid

    # Convert cartesian to polar and do transformation
    r, theta = cv.cartToPolar(x_i / x_mid, y_i / y_mid)
    # Distortion coefficients
    F3 = 0.1
    F5 = 0.12
    r = r + F3 * r ** 3 + F5 * r ** 5

    # Undo conversion, normalize and shift
    x, y = cv.polarToCart(r, theta)
    x = x * x_mid + x_mid
    y = y * y_mid + y_mid

    # Remap
    img_un = cv.remap(img,
                      x.astype(np.float32),
                      y.astype(np.float32),
                      cv.INTER_LINEAR)
