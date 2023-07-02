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
import skimage



def crop_img(img_name):

    img = cv.imread(img_name)
    rows, cols = img.shape[:2]

    # img_crop = np.zeros((int(0.08 * rows), cols), img.dtype)
    img_crop = img[int(0.3 * rows):int(0.8 * rows), int(0.2 * cols):int(0.6 * cols), :]

    plt.figure()
    plt.imshow(img_crop)
    plt.show()

    cv.imwrite("lw_4/sample_2_4.png", img_crop)


def negate_img(img_name):

    img = cv.imread(img_name)
    img_inv = 255 - img

    plt.figure()
    plt.imshow(img)
    plt.show()


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

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(img_src, cmap='gray')
    #
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(img_dst, cmap='gray')
    #
    # plt.show()

    cv.imwrite("lw_4/result_1.png", img_dst)

    return img_dst


def separate_objects(img_name):

    img_src = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    # plt.figure()
    # plt.imshow(img_src, cmap='gray')
    # plt.show()
    # return
    ret, img_dst = cv.threshold(img_src, 160, 255, cv.THRESH_BINARY_INV)
    # plt.figure()
    # plt.imshow(img_dst, cmap='gray')
    # plt.show()
    # return
    kernel_size = 7
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                        (kernel_size, kernel_size))

    # Best: kernel_size - 3; iterations - 1
    # Erosion
    bw_2 = cv.morphologyEx(img_dst,
                           cv.MORPH_ERODE,
                           kernel,
                           iterations=30,
                           borderType=cv.BORDER_CONSTANT,
                           borderValue=(0))
    # Dilation
    # plt.figure()
    # plt.imshow(bw_2, cmap='gray')
    # plt.show()
    # cv.imwrite("lw_4/result_2_1.png", bw_2)
    # return
    t = np.zeros_like(img_dst)
    while cv.countNonZero(bw_2) < bw_2.size:
        d = cv.dilate(bw_2,
                      kernel,
                      borderType=cv.BORDER_CONSTANT,
                      borderValue=(0))
        # print("Expanding")
        c = cv.morphologyEx(d,
                            cv.MORPH_CLOSE,
                            kernel,
                            borderType=cv.BORDER_CONSTANT,
                            borderValue=(0))
        s = c - d
        t = cv.bitwise_or(s, t)
        bw_2 = d
    # Closing for borders
    t = cv.morphologyEx(t,
                        cv.MORPH_CLOSE,
                        kernel,
                        iterations=31,
                        borderType=cv.BORDER_CONSTANT,
                        borderValue=(0))
    # Remove borders from an image
    t_inv = np.logical_not(t)

    t_inv = t_inv.astype(np.float32)
    img_dst = img_dst.astype(np.float32)
    img_dst = cv.bitwise_and(t_inv, img_dst)
    img_dst = np.logical_not(img_dst)

    # plt.figure()
    # plt.imshow(t_inv, cmap='gray')
    # plt.show()
    # t_inv = (255 * t_inv).astype(np.uint8)
    # cv.imwrite("lw_4/result_2_2.png", t_inv)
    # return

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src, cmap='gray')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst, cmap='gray')

    img_dst = (255 * img_dst).astype(np.uint8)
    cv.imwrite("lw_4/result_2_5.png", img_dst)

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

    cv.imwrite("lw_4/result_2_4.png", img_dst)

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

    cv.imwrite("lw_4/result_2_3.png", img_dst)

    plt.show()

    return img_dst


def bwareopen(img_bw, dim, conn=8):

    if img_bw.ndim > 2:
        return None

    # Find all connected components
    num, labels, stats, centers = cv.connectedComponentsWithStats(img_bw, connectivity=conn)

    # Check size of all connected components
    for i in range(num):

        if stats[i, cv.CC_STAT_AREA] < dim:
            img_bw[labels == i] = 0

    return img_bw


# def segment_img_old(img_name):
#
#     # Read an image
#     # Convert to grayscale and to BW
#     # Filter
#     img_src = cv.imread(img_name, cv.IMREAD_COLOR)
#     gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
#     ret, img_bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     # plt.figure()
#     # plt.imshow(img_bw, cmap='gray')
#     # plt.show()
#     # return
#     kernel_size = 5
#     # img_bw = bwareopen(img_bw, 20, 4)
#     # plt.figure()
#     # plt.imshow(img_bw, cmap='gray')
#     # plt.show()
#     # return
#     # kernel_size = 5
#     # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     # img_bw = cv.morphologyEx(img_bw,
#     #                          cv.MORPH_CLOSE,
#     #                          kernel)
#     # plt.figure()
#     # plt.imshow(img_bw, cmap='gray')
#     # plt.show()
#     # return
#
#     # Do distance transformation
#     # Find foreground location
#     # Define foreground markers
#     img_fg = cv.distanceTransform(img_bw, cv.DIST_L2, 5)
#     # plt.figure()
#     # plt.imshow(img_fg, cmap='gray')
#     # plt.show()
#     # return
#     ret, img_fg = cv.threshold(img_fg, 0.6 * img_fg.max(), 255, 0)
#     img_fg = img_fg.astype(np.uint8)
#     ret, markers = cv.connectedComponents(img_fg)
#
#     # Find background location
#     img_bg = np.zeros_like(img_bw)
#     markers_bg = markers.copy()
#     markers_bg = cv.watershed(img_src, markers_bg)
#     img_bg[markers_bg == -1] = 255
#
#     # Define undefined area
#     img_unk = cv.subtract(~img_bg, img_fg)
#
#     # Define all markers
#     markers += 1
#     markers[img_unk == 255] = 0
#
#     # Do watershed
#     # Prepare for visualization
#     markers = cv.watershed(img_src, markers)
#     markers_jet = cv.applyColorMap((markers.astype(np.float32) * 255 / (ret + 1)).astype(np.uint8),
#                                    cv.COLORMAP_JET)
#     img_src[markers == -1] = (0, 0, 255)
#
#     plt.figure()
#     plt.imshow(markers_jet)
#     plt.show()


def segment_img(img_name):

    img_src = cv.imread(img_name, cv.IMREAD_COLOR)
    gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
    ret, img_bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Do distance transformation
    # Find foreground location
    # Define foreground markers
    img_fg = cv.distanceTransform(img_bw, cv.DIST_L2, 5)
    # plt.axis('off')
    # plt.imshow(img_fg, cmap='gray')
    # plt.savefig("lw_4/result_3_1.png", bbox_inches='tight')
    # plt.show()
    # return
    ret, img_fg = cv.threshold(img_fg, 0.6 * img_fg.max(), 255, 0)
    img_fg = img_fg.astype(np.uint8)
    # plt.axis('off')
    # plt.imshow(img_fg, cmap='gray')
    # plt.savefig("lw_4/result_3_2.png", bbox_inches='tight')
    # plt.show()
    # return
    ret, markers = cv.connectedComponents(img_fg)
    # plt.axis('off')
    # plt.imshow(markers)
    # plt.savefig("lw_4/result_3_3.png", bbox_inches='tight')
    # plt.show()
    # return

    # Find background location
    img_bg = np.zeros_like(img_bw)
    markers_bg = markers.copy()
    markers_bg = cv.watershed(img_src, markers_bg)
    img_bg[markers_bg == -1] = 255
    # plt.axis('off')
    # plt.imshow(markers_bg)
    # plt.savefig("lw_4/result_3_4.png", bbox_inches='tight')
    # plt.show()
    # return
    # plt.axis('off')
    # plt.imshow(img_bg, cmap='gray')
    # plt.savefig("lw_4/result_3_5.png", bbox_inches='tight')
    # plt.show()
    # return

    # Define undefined area
    img_unk = cv.subtract(~img_bg, img_fg)
    # plt.axis('off')
    # plt.imshow(img_unk, cmap='gray')
    # plt.savefig("lw_4/result_3_6.png", bbox_inches='tight')
    # plt.show()
    # return

    # Define all markers
    markers += 1
    markers[img_unk == 255] = 0

    # Do watershed
    # Prepare for visualization
    markers = cv.watershed(img_src, markers)
    markers_jet = cv.applyColorMap((markers.astype(np.float32) * 255 / (ret + 1)).astype(np.uint8),
                                   cv.COLORMAP_JET)
    img_src[markers == -1] = (0, 0, 255)
    plt.axis('off')
    plt.imshow(markers_jet)
    plt.savefig("lw_4/result_3_7.png", bbox_inches='tight')
    plt.show()
    # return
    plt.axis('off')
    plt.imshow(img_src)
    plt.savefig("lw_4/result_3_8.png", bbox_inches='tight')
    plt.show()
    return

    plt.figure()
    plt.imshow(markers_jet)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_image')

    args = parser.parse_args()
    # crop_img(args.path_to_image)
    # negate_img(args.path_to_image)
    # remove_defects(args.path_to_image)
    # separate_objects(args.path_to_image)
    # extract_internal_borders(args.path_to_image)
    # extract_external_borders(args.path_to_image)
    segment_img(args.path_to_image)



