# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import cv2 as cv
import matplotlib.pyplot as plt
import argparse

import numpy as np


def shift_image(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    T = np.float32([[1, 0, 200], [0, 1, 400]])
    img_shifted = cv.warpAffine(img, T, (cols, rows))

    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Source Image')
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Shifted Image')
    plt.imshow(img_shifted)
    plt.savefig("pic_1.png")
    plt.show()


def reflect_image(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    T = np.float32([[-1, 0, cols - 1], [0, 1, 0]])
    img_reflected = cv.warpAffine(img, T, (cols, rows))

    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Source Image')
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Reflected Image')
    plt.imshow(img_reflected)
    plt.savefig("pic_2.png")
    plt.show()


def scale_image(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    T = np.float32([[2, 0, 0], [0, 2, 0]])
    img_scaled = cv.warpAffine(img, T, (cols, rows))

    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Source Image')
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Scaled Image')
    plt.imshow(img_scaled)
    plt.savefig("pic_3.png")
    plt.show()


def rotate_image(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    phi = 45 * math.pi / 180
    T1 = np.float32([[1, 0, -0.5 * (cols - 1)],
                     [0, 1, -0.5 * (rows - 1)],
                     [0, 0, 1]])
    T2 = np.float32([[math.cos(phi), math.sin(phi), 0],
                     [-math.sin(phi), math.cos(phi), 0],
                     [0, 0, 1]])
    T3 = np.float32([[1, 0, 0.5 * (cols - 1)],
                     [0, 1, 0.5 * (rows - 1)],
                     [0, 0, 1]])
    img_rot = cv.warpAffine(img, np.matmul(T3, np.matmul(T2, T1))[0:2, :], (cols, rows))

    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Source Image')
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Rotated Image')
    plt.imshow(img_rot)
    plt.savefig("pic_4.png")
    plt.show()


def bevel_image(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    s = 0.6
    T = np.float32([[1, s, 0],
                    [0, 1, 0]])
    img_bev = cv.warpAffine(img, T, (cols, rows))

    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Source Image')
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Beveled Image')
    plt.imshow(img_bev)
    plt.savefig("pic_5.png")
    plt.show()


def piecewise_linear_tr(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    stretch = 3
    T = np.float32([[stretch, 0, 0],
                    [0, 1, 0]])
    img_tr = img.copy()
    img_tr[:, int(cols / 2):, :] = cv.warpAffine(img_tr[:, int(cols / 2):, :],
                                                 T,
                                                 (cols - int(cols / 2), rows))

    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Source Image')
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Piecewised Image')
    plt.imshow(img_tr)
    plt.savefig("pic_6.png")
    plt.show()


def projective_tr(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    T = np.float32([[0.9, 0.2, 0.0],
                    [0.35, 0.9, -rows / 4]])
    img_tr = cv.warpAffine(img, T, (cols, rows))

    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Source Image')
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Projective Image')
    plt.imshow(img_tr)
    plt.savefig("pic_7.png")
    plt.show()


def polynomial_tr(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    T = np.float32([[0, rows / 4], [0.9, 0], [0, 0.9],
                    [0.00001, 0], [0.002, 0], [0.001, 0]])
    img_tr = np.zeros(img.shape, img.dtype)
    x, y = np.meshgrid(np.arange(cols),
                       np.arange(rows))

    der_x = T[0, 0] + x * T[1, 0] + y * T[2, 0] + x * x * T[3, 0] + x * y * T[4, 0] + y * y * T[5, 0]
    der_y = T[0, 1] + x * T[1, 1] + y * T[2, 1] + x * x * T[3, 1] + x * y * T[4, 1] + y * y * T[5, 1]
    x_new = np.round(der_x).astype(np.float32)
    y_new = np.round(der_y).astype(np.float32)

    mask = np.logical_and(np.logical_and(x_new >= 0, x_new < cols),
                          np.logical_and(y_new >= 0, y_new < rows))

    if img.ndim == 2:
        img_tr[y_new[mask].astype(int), x_new[mask].astype(int)] = img[y[mask], x[mask]]
    else:
        img_tr[y_new[mask].astype(int), x_new[mask].astype(int), :] = img[y[mask], x[mask], :]

    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Source Image')
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Polynomial Image')
    plt.imshow(img_tr)
    plt.savefig("pic_8.png")
    plt.show()


def harmonic_tr(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x + 10 * np.sin(2 * math.pi * y / 90)
    img_tr = cv.remap(img,
                      x.astype(np.float32),
                      y.astype(np.float32),
                      cv.INTER_LINEAR)

    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Source Image')
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Sinusoidal Image')
    plt.imshow(img_tr)
    plt.savefig("pic_9.png")
    plt.show()


def undistort_img(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]

    x_i, y_i = np.meshgrid(np.arange(cols), np.arange(rows))
    x_mid = cols / 2.0
    y_mid = rows / 2.0
    x_i = x_i - x_mid
    y_i = y_i - y_mid

    r, theta = cv.cartToPolar(x_i / x_mid, y_i / y_mid)
    F3 = 0.1
    F5 = 0.12
    r = r + F3 * r ** 3 + F5 * r ** 5

    x, y = cv.polarToCart(r, theta)
    x = x * x_mid + x_mid
    y = y * y_mid + y_mid

    img_un = cv.remap(img,
                      x.astype(np.float32),
                      y.astype(np.float32),
                      cv.INTER_LINEAR)

    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Source Image')
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Undistorted Image')
    plt.imshow(img_un)
    plt.savefig("pic_10.png")
    plt.show()


def crop_img(img_name):

    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]

    img_1 = np.zeros((rows, int(0.35 * cols)), img.dtype)
    img_2 = np.zeros((rows, int(0.25 * cols)), img.dtype)
    img_1 = img[:, :int(0.35 * cols), :]
    img_2 = img[:, int(0.25 * cols):, :]

    cv.imwrite("sample_3_1.png", img_1)
    cv.imwrite("sample_3_2.png", img_2)


def stitch_images(img_left_name, img_right_name):

    img_left = cv.imread(img_left_name)
    img_right = cv.imread(img_right_name)

    templ_size = 10
    templ = img_left[-templ_size:, :, :]
    res = cv.matchTemplate(img_right, templ, cv.TM_CCOEFF)

    _, _, _, max_loc = cv.minMaxLoc(res)

    img_dst = np.zeros((img_left.shape[0],
                        img_left.shape[1] + img_right.shape[1] - max_loc[1] - templ_size,
                        img_left.shape[2]),
                       dtype=np.uint8)

    img_dst[:, 0:img_left.shape[1], :] = img_left
    print(f'Shape left img {img_left.shape}')
    print(f'Shape right img {img_right.shape}')
    print(f'max_loc[1] {max_loc[1]}')
    print(f'Shape img_right[:, max_loc[1] + templ_size, :] {img_right[:, max_loc[1] + templ_size, :].shape}')
    img_dst[:, img_left.shape[1]:, :] = img_right[:, max_loc[1] + templ_size:, :]

    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(img_left)

    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(img_right)

    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(img_dst)
    plt.savefig("pic_11.png")
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_image')
    parser.add_argument('--path_to_left_image')
    parser.add_argument('--path_to_right_image')

    args = parser.parse_args()

    stitch_images(args.path_to_left_image,
                  args.path_to_right_image)
