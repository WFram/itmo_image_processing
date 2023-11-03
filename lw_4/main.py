# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import cv2 as cv
import matplotlib.pyplot as plt
import argparse

import numpy as np


def crop_img(img_name):

    img = cv.imread(img_name)
    rows, cols = img.shape[:2]

    img_crop = img[int(0.3 * rows):int(0.8 * rows), int(0.2 * cols):int(0.6 * cols), :]

    plt.figure()
    plt.imshow(img_crop)
    plt.show()

    cv.imwrite("sample_2_4.png", img_crop)


def remove_defects(img_name):

    img_src = cv.imread(img_name)
    img_src = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)

    a_kernel = 3
    b_kernel = 6
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (a_kernel, b_kernel))
    img_dst = cv.morphologyEx(img_src,
                              cv.MORPH_OPEN,
                              kernel,
                              iterations=35,
                              borderType=cv.BORDER_CONSTANT,
                              borderValue=(255))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (b_kernel, a_kernel))
    img_dst = cv.morphologyEx(img_dst,
                              cv.MORPH_OPEN,
                              kernel,
                              iterations=55,
                              borderType=cv.BORDER_CONSTANT,
                              borderValue=(255))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (a_kernel, a_kernel))
    img_dst = cv.morphologyEx(img_dst,
                              cv.MORPH_ERODE,
                              kernel,
                              iterations=28,
                              borderType=cv.BORDER_CONSTANT,
                              borderValue=(255))

    cv.imwrite("result_1.png", img_dst)

    return img_dst


def separate_objects(img_name):

    img_src = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    _, img_dst = cv.threshold(img_src, 160, 255, cv.THRESH_BINARY_INV)
    kernel_size = 7
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                        (kernel_size, kernel_size))

    bw_2 = cv.morphologyEx(img_dst,
                           cv.MORPH_ERODE,
                           kernel,
                           iterations=30,
                           borderType=cv.BORDER_CONSTANT,
                           borderValue=(0))
    
    t = np.zeros_like(img_dst)
    while cv.countNonZero(bw_2) < bw_2.size:
        d = cv.dilate(bw_2,
                      kernel,
                      borderType=cv.BORDER_CONSTANT,
                      borderValue=(0))

        c = cv.morphologyEx(d,
                            cv.MORPH_CLOSE,
                            kernel,
                            borderType=cv.BORDER_CONSTANT,
                            borderValue=(0))
        s = c - d
        t = cv.bitwise_or(s, t)
        bw_2 = d

    t = cv.morphologyEx(t,
                        cv.MORPH_CLOSE,
                        kernel,
                        iterations=31,
                        borderType=cv.BORDER_CONSTANT,
                        borderValue=(0))
 
    t_inv = np.logical_not(t)

    t_inv = t_inv.astype(np.float32)
    img_dst = img_dst.astype(np.float32)
    img_dst = cv.bitwise_and(t_inv, img_dst)
    img_dst = np.logical_not(img_dst)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src, cmap='gray')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst, cmap='gray')

    img_dst = (255 * img_dst).astype(np.uint8)
    cv.imwrite("result_2_5.png", img_dst)

    plt.show()

    return img_dst


def extract_internal_borders(img_name):

    img_src = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    _, img_dst = cv.threshold(img_src, 127, 255, cv.THRESH_BINARY_INV)
    _, img_src = cv.threshold(img_src, 127, 255, cv.THRESH_BINARY_INV)

    kernel_size = 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                      (kernel_size, kernel_size))

    bw_2 = cv.morphologyEx(img_dst,
                           cv.MORPH_ERODE,
                           kernel,
                           iterations=3,
                           borderType=cv.BORDER_CONSTANT,
                           borderValue=(0))

    img_dst = img_src - bw_2

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src, cmap='gray')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst, cmap='gray')

    cv.imwrite("result_2_4.png", img_dst)

    plt.show()

    return img_dst


def extract_external_borders(img_name):

    img_src = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    _, img_dst = cv.threshold(img_src, 127, 255, cv.THRESH_BINARY_INV)
    _, img_src = cv.threshold(img_src, 127, 255, cv.THRESH_BINARY_INV)

    kernel_size = 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                      (kernel_size, kernel_size))

    bw_2 = cv.morphologyEx(img_dst,
                           cv.MORPH_DILATE,
                           kernel,
                           iterations=3,
                           borderType=cv.BORDER_CONSTANT,
                           borderValue=(0))

    img_dst = bw_2 - img_src

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src, cmap='gray')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst, cmap='gray')

    cv.imwrite("result_2_3.png", img_dst)

    plt.show()

    return img_dst


def bwareopen(img_bw, dim, conn=8):

    if img_bw.ndim > 2:
        return None

    num, labels, stats, centers = cv.connectedComponentsWithStats(img_bw, connectivity=conn)

    for i in range(num):

        if stats[i, cv.CC_STAT_AREA] < dim:
            img_bw[labels == i] = 0

    return img_bw


def segment_img(img_name):

    img_src = cv.imread(img_name, cv.IMREAD_COLOR)
    gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
    ret, img_bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    img_fg = cv.distanceTransform(img_bw, cv.DIST_L2, 5)
    
    ret, img_fg = cv.threshold(img_fg, 0.6 * img_fg.max(), 255, 0)
    img_fg = img_fg.astype(np.uint8)
    
    ret, markers = cv.connectedComponents(img_fg)
    
    img_bg = np.zeros_like(img_bw)
    markers_bg = markers.copy()
    markers_bg = cv.watershed(img_src, markers_bg)
    img_bg[markers_bg == -1] = 255
    
    img_unk = cv.subtract(~img_bg, img_fg)
    
    markers += 1
    markers[img_unk == 255] = 0

    markers = cv.watershed(img_src, markers)
    markers_jet = cv.applyColorMap((markers.astype(np.float32) * 255 / (ret + 1)).astype(np.uint8),
                                   cv.COLORMAP_JET)
    img_src[markers == -1] = (0, 0, 255)
    plt.axis('off')
    plt.imshow(markers_jet)
    plt.savefig("result_3_7.png", bbox_inches='tight')
    plt.show()
    
    plt.axis('off')
    plt.imshow(img_src)
    plt.savefig("result_3_8.png", bbox_inches='tight')
    plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_image')

    args = parser.parse_args()

    segment_img(args.path_to_image)
