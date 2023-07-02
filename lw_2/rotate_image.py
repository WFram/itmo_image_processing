import cv2 as cv
import numpy as np
import math


def rotate_image(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    # Define the rotation angle
    phi = z * math.pi / 180
    # Straight shift to center an image
    T1 = np.float32([[1, 0, -0.5 * (cols - 1)],
                     [0, 1, -0.5 * (rows - 1)],
                     [0, 0, 1]])
    # Rotation
    T2 = np.float32([[math.cos(phi), math.sin(phi), 0],
                     [-math.sin(phi), math.cos(phi), 0],
                     [0, 0, 1]])
    # Reverse shift
    T3 = np.float32([[1, 0, 0.5 * (cols - 1)],
                     [0, 1, 0.5 * (rows - 1)],
                     [0, 0, 1]])
    img_rot = cv.warpAffine(img, np.matmul(T3, np.matmul(T2, T1))[0:2, :], (cols, rows))
