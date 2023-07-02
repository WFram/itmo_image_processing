import numpy as np
import cv2 as cv


def calc_projection(img_file):

    img = cv.imread(img_file)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Calculate projection to Oy
    proj_y = np.sum(img, 1) / 255 if img.ndim == 2 else np.sum(img, (1, 2)) / 255 / img.shape[2]
    # Calculate projection to Ox
    proj_x = np.sum(img, 0) / 255 if img.ndim == 2 else np.sum(img, (0, 2)) / 255 / img.shape[2]

    # Find outlines
    get_object_outlines(proj_x, tol=1e-1)
