# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.transforms as tr
import os
import sys
import argparse

import numpy as np


def calc_hists_plt(img, img_new, hist_name):

    hist_size = 256
    hist_range = [0, 256]

    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]

    img_new_b = img_new[:, :, 0]
    img_new_g = img_new[:, :, 1]
    img_new_r = img_new[:, :, 2]

    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(22)
    ax = fig.add_subplot(2, 4, 1)
    plt.hist(img_b.ravel(), hist_size, hist_range)
    ax.set_title('BLUE')
    ax.set_ylim([0, 20000])
    ax.set_xlim([0, 255])

    ax = fig.add_subplot(2, 4, 2)
    plt.hist(img_g.ravel(), hist_size, hist_range)
    ax.set_title('GREEN')
    ax.set_ylim([0, 20000])
    ax.set_xlim([0, 255])

    ax = fig.add_subplot(2, 4, 3)
    plt.hist(img_r.ravel(), hist_size, hist_range)
    ax.set_title('RED')
    ax.set_ylim([0, 20000])
    ax.set_xlim([0, 255])

    ax = fig.add_subplot(2, 4, 4)
    plt.imshow(img)
    ax.set_title('Source Image')

    ax = fig.add_subplot(2, 4, 5)
    plt.hist(img_new_b.ravel(), hist_size, hist_range)
    ax.set_title('BLUE')
    ax.set_ylim([0, 20000])
    ax.set_xlim([0, 255])

    ax = fig.add_subplot(2, 4, 6)
    plt.hist(img_new_g.ravel(), hist_size, hist_range)
    ax.set_title('GREEN')
    ax.set_ylim([0, 20000])
    ax.set_xlim([0, 255])

    ax = fig.add_subplot(2, 4, 7)
    plt.hist(img_new_r.ravel(), hist_size, hist_range)
    ax.set_title('RED')
    ax.set_ylim([0, 20000])
    ax.set_xlim([0, 255])

    ax = fig.add_subplot(2, 4, 8)
    plt.imshow(img_new)
    ax.set_title('Final Image')

    plt.savefig(hist_name)
    plt.show()


# For grayscale image (one layer)
def calc_hist_cv(img):

    hist_size = 256
    hist_range = (0, 256)

    hist = []
    for i in range(3):
        hist.append(cv.calcHist(img, [i], None, [hist_size], hist_range))

    return hist


def shift_hists(img_file, shift_val, show_hists):

    img = cv.imread(img_file)
    img_new = np.copy(img)

    LUT = []

    for i in range(256):
        LUT.append(255) if i + shift_val > 255 else LUT.append(i + shift_val)

    LUT = np.array(LUT, dtype=np.uint8)
    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]
    img_b = LUT[img_b]
    img_g = LUT[img_g]
    img_r = LUT[img_r]
    img_new[:, :, 0] = img_b
    img_new[:, :, 1] = img_g
    img_new[:, :, 2] = img_r

    cv.imwrite("sample_1_2.png", img_new)

    plt.figure()
    plt.imshow(img)
    plt.show()

    if show_hists:
        calc_hists_plt(img, img_new, "pic_2.png")


def stretch_nonlinear(img_file, alpha, show_hists):

    img = cv.imread(img_file)

    img_new = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_bgr = cv.split(img_new)
    img_new_bgr = []

    for layer in img_bgr:

        img_min = layer.min()
        img_max = layer.max()
        img_new = np.clip((((layer - img_min) / (img_max - img_min)) ** alpha), 0, 1)
        img_new_bgr.append(img_new)

    img_new = cv.merge(img_new_bgr)

    if img.dtype == np.uint8:
        img_new = (255 * img_new).clip(0, 255).astype(np.uint8)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_3.png")

    return img_new


def stretch_uniform(img_file, show_hists):

    img = cv.imread(img_file)

    img_new = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_bgr = cv.split(img)
    img_new_bgr = cv.split(img_new)
    img_final = []

    hist = calc_hist_cv(img_bgr)
    for i, layer in enumerate(img_new_bgr):
        img_min = layer.min()
        img_max = layer.max()
        cum_hist = (255 * (np.cumsum(hist[i])) / (layer.shape[0] * layer.shape[1])).clip(0, 255).astype(np.uint8)
        layer = (255 * layer).clip(0, 255).astype(np.uint8)
        img_new = np.clip(((img_max - img_min) * (cum_hist[layer] / 255) + img_min), 0, 1)
        img_final.append(img_new)

    img_new = cv.merge(img_final)

    if img.dtype == np.uint8:
        img_new = (255 * img_new).clip(0, 255).astype(np.uint8)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_4.png")

    return img_new


def exponential_conversion(img_file, alpha, show_hists):

    img = cv.imread(img_file)

    img_new = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_bgr = cv.split(img)
    img_new_bgr = cv.split(img_new)
    img_final = []

    hist = calc_hist_cv(img_bgr)
    inv_alpha = 1 / alpha
    for i, layer in enumerate(img_new_bgr):
        img_min = layer.min()
        cum_hist = (255 * (np.cumsum(hist[i])) / (layer.shape[0] * layer.shape[1])).clip(0, 255).astype(np.uint8)
        layer = (255 * layer).clip(0, 255).astype(np.uint8)
        cum_hist = (cum_hist.astype(np.float32) / 255).clip(0, 1)
        img_new = np.clip((img_min - inv_alpha * (np.log(1 - cum_hist[layer] + 0.001))), 0, 1)
        img_final.append(img_new)

    img_new = cv.merge(img_final)

    if img.dtype == np.uint8:
        img_new = (255 * img_new).clip(0, 255).astype(np.uint8)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_5.png")

    return img_new


def rayleigh_conversion(img_file, alpha, show_hists):

    img = cv.imread(img_file)

    img_new = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_bgr = cv.split(img)
    img_new_bgr = cv.split(img_new)
    img_final = []

    hist = calc_hist_cv(img_bgr)
    inv_alpha = 1 / alpha
    for i, layer in enumerate(img_new_bgr):
        img_min = layer.min()
        cum_hist = (255 * (np.cumsum(hist[i])) / (layer.shape[0] * layer.shape[1])).clip(0, 255).astype(np.uint8)
        layer = (255 * layer).clip(0, 255).astype(np.uint8)
        cum_hist = (cum_hist.astype(np.float32) / 255).clip(0, 1)
        beta = np.log(1 / (1 - cum_hist[layer] + 0.001))
        epsilon = np.sqrt(2 * alpha ** 2 * beta + 0.001)
        img_new = np.clip((img_min + epsilon), 0, 1)
        img_final.append(img_new)

    img_new = cv.merge(img_final)

    if img.dtype == np.uint8:
        img_new = (255 * img_new).clip(0, 255).astype(np.uint8)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_6.png")

    return img_new


def law_2_3(img_file, show_hists):

    img = cv.imread(img_file)

    img_new = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_bgr = cv.split(img)
    img_new_bgr = cv.split(img_new)
    img_final = []

    hist = calc_hist_cv(img_bgr)
    for i, layer in enumerate(img_new_bgr):
        cum_hist = (255 * (np.cumsum(hist[i])) / (layer.shape[0] * layer.shape[1])).clip(0, 255).astype(np.uint8)
        layer = (255 * layer).clip(0, 255).astype(np.uint8)
        cum_hist = (cum_hist.astype(np.float32) / 255).clip(0, 1)
        img_new = np.clip((cum_hist[layer] ** (2 / 3)), 0, 1)
        img_final.append(img_new)

    img_new = cv.merge(img_final)

    if img.dtype == np.uint8:
        img_new = (255 * img_new).clip(0, 255).astype(np.uint8)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_7.png")

    return img_new


def hyperbolic_conversion(img_file, alpha, show_hists):

    img = cv.imread(img_file)

    img_new = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_bgr = cv.split(img)
    img_new_bgr = cv.split(img_new)
    img_final = []

    hist = calc_hist_cv(img_bgr)
    for i, layer in enumerate(img_new_bgr):
        cum_hist = (255 * (np.cumsum(hist[i])) / (layer.shape[0] * layer.shape[1])).clip(0, 255).astype(np.uint8)
        layer = (255 * layer).clip(0, 255).astype(np.uint8)
        cum_hist = (cum_hist.astype(np.float32) / 255).clip(0, 1)
        img_new = np.clip((alpha ** cum_hist[layer]), 0, 1)
        img_final.append(img_new)

    img_new = cv.merge(img_final)

    if img.dtype == np.uint8:
        img_new = (255 * img_new).clip(0, 255).astype(np.uint8)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_8.png")

    return img_new


def equalize(img_file, show_hists):

    img = cv.imread(img_file)

    img_bgr = cv.split(img)
    img_final = []

    for i, layer in enumerate(img_bgr):
        img_new = np.clip((cv.equalizeHist(layer)), 0, 255)
        img_final.append(img_new)

    img_new = cv.merge(img_final)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_9.png")

    return img_new


def clahe_equalize(img_file, show_hists):

    img = cv.imread(img_file)

    img_bgr = cv.split(img)
    img_final = []

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i, layer in enumerate(img_bgr):
        img_new = np.clip((clahe.apply(layer)), 0, 255)
        img_final.append(img_new)

    img_new = cv.merge(img_final)

    if show_hists:
        calc_hists_plt(img, img_new, "pic_10.png")

    return img_new


def get_comparison(img_file,
                   alpha_stretch_nonlinear,
                   alpha_exponential_conversion=5,
                   alpha_rayleigh_conversion=0.2,
                   alpha_hyperbolic_conversion=0.05,
                   show_hists=False):

    img = cv.imread(img_file)

    img_stretch_nonlinear = stretch_nonlinear(img_file, alpha=alpha_stretch_nonlinear, show_hists=show_hists)
    img_stretch_uniform = stretch_uniform(img_file, show_hists=show_hists)
    img_exponential_conversion = exponential_conversion(img_file, alpha=alpha_exponential_conversion, show_hists=show_hists)
    img_rayleigh_conversion = rayleigh_conversion(img_file, alpha=alpha_rayleigh_conversion, show_hists=show_hists)
    img_law_2_3 = law_2_3(img_file, show_hists=show_hists)
    img_hyperbolic_conversion = hyperbolic_conversion(img_file, alpha=alpha_hyperbolic_conversion, show_hists=show_hists)
    img_equalize = equalize(img_file, show_hists=show_hists)
    img_clahe_equalize = clahe_equalize(img_file, show_hists=show_hists)

    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(22)
    ax = fig.add_subplot(3, 3, 1)
    plt.imshow(img)
    ax.set_title("Source Image")

    ax = fig.add_subplot(3, 3, 2)
    plt.imshow(img_stretch_nonlinear)
    ax.set_title("Nonlinear Stretch")

    ax = fig.add_subplot(3, 3, 3)
    plt.imshow(img_stretch_uniform)
    ax.set_title("Uniform Stretch")

    ax = fig.add_subplot(3, 3, 4)
    plt.imshow(img_exponential_conversion)
    ax.set_title("Exponential Conversion")

    ax = fig.add_subplot(3, 3, 5)
    plt.imshow(img_rayleigh_conversion)
    ax.set_title("Rayleigh Law Conversion")

    ax = fig.add_subplot(3, 3, 6)
    plt.imshow(img_law_2_3)
    ax.set_title("2/3 Law Conversion")

    ax = fig.add_subplot(3, 3, 7)
    plt.imshow(img_hyperbolic_conversion)
    ax.set_title("Hyperbolic Conversion")

    ax = fig.add_subplot(3, 3, 8)
    plt.imshow(img_equalize)
    ax.set_title("Equalization")

    ax = fig.add_subplot(3, 3, 9)
    plt.imshow(img_clahe_equalize)
    ax.set_title("CLAHE")
    plt.savefig("comparison.png")
    plt.show()


def calc_profile(img_file):

    img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
    x_profile = img[round(img.shape[0] / 2), :]

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plt.imshow(img, cmap='gray')

    ax = fig.add_subplot(2, 1, 2)
    plt.plot(x_profile, color='black')
    plt.savefig("pic_11.png")
    plt.show()


def get_object_boards(proj, tol):

    init_val = proj[0]
    low_found = False
    high_found = False
    for i, val in enumerate(proj):

        if math.fabs(init_val - val) >= tol and not low_found:
            low_found = True
            high_found = False
            print(f'Low boarder: {i}')
        if math.fabs(init_val - val) < tol and low_found and not high_found:
            high_found = True
            low_found = False
            print(f'High boarder: {i - 1}')


def calc_projection(img_file):

    img = cv.imread(img_file)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    proj_y = np.sum(img, 1) / 255 if img.ndim == 2 else np.sum(img, (1, 2)) / 255 / img.shape[2]
    proj_x = np.sum(img, 0) / 255 if img.ndim == 2 else np.sum(img, (0, 2)) / 255 / img.shape[2]

    proj_y_new = np.empty((proj_y.shape[0], 2))
    proj_y_new[:, 1] = proj_y[:]
    proj_y_new[:, 0] = range(0, proj_y.shape[0])

    proj_x_new = np.empty((proj_x.shape[0], 2))
    proj_x_new[:, 0] = proj_x[:]
    proj_x_new[:, 1] = range(0, proj_x.shape[0])
    # print(proj_y_new[:, 1])

    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(10)
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Source Image")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(0, img.shape[0])
    plt.imshow(img)

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("Oy Projection")
    ax.set_ylim(0, img.shape[0])
    plt.plot(proj_y_new[:, 1], proj_y_new[:, 0])

    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("Ox Projection")
    ax.set_xlim(0, img.shape[1])
    plt.plot(proj_x_new[:, 1], proj_x_new[:, 0])
    plt.gca().invert_yaxis()
    plt.savefig("pic_12.png")
    plt.show()

    get_object_boards(proj_y, tol=1e-1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_image')
    args = parser.parse_args()

    # shift_hists(args.path_to_image, shift_val=100, show_hists=True)
    # stretch_nonlinear(args.path_to_image, alpha=1.1, show_hists=True)
    # stretch_uniform(args.path_to_image, show_hists=True)
    # exponential_conversion(args.path_to_image, alpha=5, show_hists=True)
    # rayleigh_conversion(args.path_to_image, alpha=0.2, show_hists=True)
    # law_2_3(args.path_to_image, show_hists=True)
    # hyperbolic_conversion(args.path_to_image, alpha=0.05, show_hists=True)
    # equalize(args.path_to_image, show_hists=True)
    # clahe_equalize(args.path_to_image, show_hists=True)
    # get_comparison(args.path_to_image,
    #                alpha_stretch_nonlinear=1.1,
    #                alpha_exponential_conversion=5,
    #                alpha_rayleigh_conversion=0.2,
    #                alpha_hyperbolic_conversion=0.05,
    #                show_hists=False)
    calc_projection(args.path_to_image)
