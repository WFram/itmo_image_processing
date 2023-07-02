import cv2 as cv
import numpy as np
import math


def harmonic_tr(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x + 10 * np.sin(2 * math.pi * y / 90)
    # Transform according to new coordinates
    img_tr = cv.remap(img,
                      x.astype(np.float32),
                      y.astype(np.float32),
                      cv.INTER_LINEAR)
