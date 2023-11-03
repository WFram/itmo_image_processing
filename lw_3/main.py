# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import cv2 as cv
import matplotlib.pyplot as plt
import argparse

import numpy as np
import skimage


def apply_salt_n_pepper_noise(img_name, amount=0.4):
    img = cv.imread(img_name)

    img_noise = skimage.util.random_noise(img, 's&p', amount=amount)
    img_noise = (255 * img_noise).clip(0, 255).astype(np.uint8)

    cv.imwrite('sample_1_1_3.png', img_noise)

    return img_noise


def apply_pepper_noise(img_name):
    img = cv.imread(img_name)

    img_noise = skimage.util.random_noise(img, 'pepper', amount=0.4)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_noise)

    plt.show()


def apply_salt_noise(img_name):
    img = cv.imread(img_name)

    img_noise = skimage.util.random_noise(img, 'salt', amount=0.4)
    img_noise = (255 * img_noise).clip(0, 255).astype(np.uint8)

    cv.imwrite('sample_1_1_1.png', img_noise)

    plt.show()


def apply_speckle_noise(img_name, mean=2, var=0.2):

    img = cv.imread(img_name)

    img_noise = skimage.util.random_noise(img, 'speckle', mean=mean, var=var)
    img_noise = (255 * img_noise).clip(0, 255).astype(np.uint8)

    cv.imwrite('sample_1_3.png', img_noise)

    return img_noise


def apply_additive_noise(img_name, mean=0, var=0.4):

    img = cv.imread(img_name)

    img_noise = skimage.util.random_noise(img, 'additive', mean=mean, var=var)
    img_noise = (255 * img_noise).clip(0, 255).astype(np.uint8)

    cv.imwrite('sample_1_2.png', img_noise)

    return img_noise


def apply_gaussian_noise(img_name, mean=5, var=0.4, use_local_var=False):
    img = cv.imread(img_name)

    if use_local_var:
        xn = np.array(mean)
        xn = xn.astype(np.float32)
        xn /= float(np.iinfo(np.uint8).max)
        local_var = skimage.filters.gaussian(xn, sigma=var) + 1e-10
        img_noise = skimage.util.random_noise(img, 'localvar', local_vars=local_var * 0.5)
    else:
        img_noise = skimage.util.random_noise(img, 'gaussian', mean=mean, var=var)

    img_noise = (255 * img_noise).clip(0, 255).astype(np.uint8)

    cv.imwrite('sample_1_4.png', img_noise)

    return img_noise


def apply_poisson_noise(img_name, peak=5.0):
    img = cv.imread(img_name)
    img_new = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img

    noise_mask = np.random.poisson(img_new * peak)
    img_noise = img_new + noise_mask
    if img.dtype == np.uint8:
        img_noise = (255 * img_noise).clip(0, 255).astype(np.uint8)

    cv.imwrite('sample_1_5.png', img_noise)

    return img_noise


def get_comparison(img_name):
    img = cv.imread(img_name)
    img_snp = apply_salt_n_pepper_noise(img_name, amount=0.4)
    img_speckle = apply_speckle_noise(img_name, mean=1, var=0.2)
    img_additive = apply_additive_noise(img_name, mean=0, var=0.8)
    img_gauss = apply_gaussian_noise(img_name, mean=0, var=0.07, use_local_var=False)
    img_poisson = apply_poisson_noise(img_name, peak=4.0)

    fig = plt.figure()
    fig.set_figheight(11)
    fig.set_figwidth(10)
    ax = fig.add_subplot(3, 2, 1)
    plt.imshow(img)
    ax.set_title("Source Image")

    ax = fig.add_subplot(3, 2, 2)
    plt.imshow(img_snp)
    ax.set_title("Salt and Pepper")

    ax = fig.add_subplot(3, 2, 3)
    plt.imshow(img_additive)
    ax.set_title("Additive")

    ax = fig.add_subplot(3, 2, 4)
    plt.imshow(img_speckle)
    ax.set_title("Speckle")

    ax = fig.add_subplot(3, 2, 5)
    plt.imshow(img_gauss)
    ax.set_title("Gaussian")

    ax = fig.add_subplot(3, 2, 6)
    plt.imshow(img_poisson)
    ax.set_title("Poisson")

    plt.savefig("comparison.png")
    plt.show()


def filter_ar_average(img_name, kernel_size):
    img_src = cv.imread(img_name)

    img_dst = cv.blur(img_src, kernel_size)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def filter_geom_average(img_name, kernel_size):

    img_src = cv.imread(img_name)
    kernel = np.ones(kernel_size)
    rows, cols = img_src.shape[:2]

    img_copy = img_src.astype(np.float32) / 255 if img_src.dtype == np.uint8 else img_src
    img_copy = cv.copyMakeBorder(img_copy,
                                 int((kernel_size[0] - 1) / 2),
                                 int(kernel_size[0] / 2),
                                 int((kernel_size[1] - 1) / 2),
                                 int(kernel_size[1] / 2),
                                 cv.BORDER_REPLICATE)

    layers = cv.split(img_copy)
    layers_new = []
    for layer in layers:

        m = np.ones(img_src.shape[:2], np.float32)

        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                temp = m * kernel[i, j]
                m = temp * layer[i:i + rows, j:j + cols]

        power = 1 / (kernel_size[0] * kernel_size[1])
        layer_new = np.power(m, power)
        layers_new.append(layer_new)

    img_dst = cv.merge(layers_new)

    if img_src.dtype == np.uint8:
        img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def filter_counter_harmonic_average(img_name, kernel_size=(3, 3), Q=0):

    img_src = cv.imread(img_name)
    kernel = np.ones(kernel_size)
    kernel /= kernel_size[0] / kernel_size[1]
    rows, cols = img_src.shape[:2]

    img_copy = img_src.astype(np.float32) / 255 if img_src.dtype == np.uint8 else img_src
    img_copy = cv.copyMakeBorder(img_copy,
                                 int((kernel_size[0] - 1) / 2),
                                 int(kernel_size[0] / 2),
                                 int((kernel_size[1] - 1) / 2),
                                 int(kernel_size[1] / 2),
                                 cv.BORDER_REPLICATE)

    layers = cv.split(img_copy)
    layers_new = []
    for layer in layers:

        m = np.zeros(img_src.shape[:2], np.float32)
        q = np.zeros(img_src.shape[:2], np.float32)
        n = np.power(layer, Q)
        p = np.power(layer, Q + 1)

        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                m = m + kernel[i, j] * p[i:i + rows, j:j + cols]
                q = q + kernel[i, j] * n[i:i + rows, j:j + cols]

        layer_new = m / q
        layers_new.append(layer_new)

    img_dst = cv.merge(layers_new)

    if img_src.dtype == np.uint8:
        img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def filter_gaussian(img_name, kernel=7, sigma=0.0):
    img_src = cv.imread(img_name)

    img_dst = cv.GaussianBlur(img_src, (kernel, kernel), sigma)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    return img_dst


def filter_median(img_name):
    img_src = cv.imread(img_name)

    kernel = 7
    img_dst = cv.medianBlur(img_src, kernel)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def filter_weighted_rank(img_name, kernel_size, rank=4, use_median=False, use_weighted=False):
    img = cv.imread(img_name)

    kernel = np.ones(kernel_size, dtype=np.float32)
    if use_median:
        N = kernel_size[0] * kernel_size[1]
        rank = int(0.5 * (N - 1))
        print(f"Rank: {rank}")
    if use_weighted:
        kernel = np.array([[0.6, 0.7, 0.8, 0.7, 0.6],
                           [0.7, 0.8, 0.9, 0.8, 0.7],
                           [0.8, 0.9, 1.0, 0.9, 0.8],
                           [0.7, 0.8, 0.9, 0.8, 0.7],
                           [0.6, 0.7, 0.8, 0.7, 0.6]], dtype=np.float32)
    rows, cols = img.shape[:2]

    img_copy = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_copy = cv.copyMakeBorder(img_copy,
                                 int((kernel_size[0] - 1) / 2),
                                 int(kernel_size[0] / 2),
                                 int((kernel_size[1] - 1) / 2),
                                 int(kernel_size[1] / 2),
                                 cv.BORDER_REPLICATE)

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

    img_layers.sort()

    img_dst = img_layers[:, :, rank] if img.ndim == 2 else img_layers[:, :, :, rank]

    if img.dtype == np.uint8:
        img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def adaptive_filter_iterate(img, i, j, kernel_size, s_max):

    coords = []
    for m in range(-int(kernel_size / 2), int(kernel_size / 2) + 1):
        for n in range(-int(kernel_size / 2), int(kernel_size / 2) + 1):
            coords.append(img[i + m, j + n, 0])

    coords.sort()
    z_min = coords[0]
    z_max = coords[-1]
    z_max = coords[int(kernel_size * kernel_size - 1)]
    z_med = coords[int(kernel_size * kernel_size / 2)]
    z_xy = img[i, j, 0]

    ret_val = z_med
    if z_med > z_min and z_med < z_max:
        ret_val = z_xy if z_xy > z_min and z_xy < z_max else z_med
    else:
        kernel_size += 2
        ret_val = adaptive_filter_iterate(img, i, j, kernel_size, s_max) if kernel_size <= s_max else z_med

    return ret_val


def filter_adaptive_median(img_name, min_kernel_size=3, max_kernel_size=5):

    img_src = cv.imread(img_name)

    s_min = min_kernel_size
    s_max = max_kernel_size

    img_copy = img_src.astype(np.float32) / 255 if img_src.dtype == np.uint8 else img_src
    img_copy = cv.copyMakeBorder(img_copy,
                                 int((s_min / 2) - 1),
                                 int(s_max),
                                 int((s_min / 2) - 1),
                                 int(s_max),
                                 cv.BORDER_REPLICATE)
    rows, cols = img_copy.shape[:2]
    for i in range(int(s_max / 2), rows - int(s_max / 2)):
        for j in range(int(s_max / 2), cols - int(s_max / 2)):
            img_copy[i, j] = adaptive_filter_iterate(img_copy, i, j, s_min, s_max)

    img_dst = np.copy(img_copy)

    if img_src.dtype == np.uint8:
        img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def filter_wiener(img_name, kernel_size=(9, 9)):

    img = cv.imread(img_name)

    kernel = np.ones(kernel_size)
    rows, cols = img.shape[:2]

    img_copy = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    img_copy = cv.copyMakeBorder(img_copy,
                                 int((kernel_size[0] - 1) / 2),
                                 int(kernel_size[0] / 2),
                                 int((kernel_size[1] - 1) / 2),
                                 int(kernel_size[1] / 2),
                                 cv.BORDER_REPLICATE)

    layers = cv.split(img_copy)
    layers_new = []
    k_power = np.power(kernel, 2)
    for layer in layers:

        layer_power = np.power(layer, 2)
        m = np.zeros(img.shape[:2], np.float32)
        q = np.zeros(img.shape[:2], np.float32)

        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                m = m + kernel[i, j] * layer[i:i + rows, j:j + cols]
                q = q + k_power[i, j] * layer_power[i:i + rows, j:j + cols]

        m /= np.sum(kernel)
        q /= np.sum(kernel)
        q = q - m * m

        v = np.sum(q) / img.size

        layer_new = layer[(kernel_size[0] - 1) // 2:(kernel_size[0] - 1) // 2 + rows,
                    (kernel_size[1] - 1) // 2:(kernel_size[1] - 1) // 2 + cols]
        layer_new = np.where(q < v, m, (layer_new - m) * (1 - v / q) + m)
        layers_new.append(layer_new)

    img_dst = cv.merge(layers_new)

    if img.dtype == np.uint8:
        img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def filter_roberts(img_name):

    img_src = cv.imread(img_name)
    img_src = img_src.astype(np.float32) / 255

    g_x = np.array([[1, -1],
                    [0, 0]])
    g_y = np.array([[1, 0],
                    [-1, 0]])
    i_x = cv.filter2D(img_src, -1, g_x)
    i_y = cv.filter2D(img_src, -1, g_y)
    img_dst = cv.magnitude(i_x, i_y)

    img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def filter_prewitt(img_name):

    img_src = cv.imread(img_name)
    img_src = img_src.astype(np.float32) / 255

    g_x = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])
    g_y = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])
    i_x = cv.filter2D(img_src, -1, g_x)
    i_y = cv.filter2D(img_src, -1, g_y)
    img_dst = cv.magnitude(i_x, i_y)

    img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def filter_sobel(img_name):

    img_src = cv.imread(img_name)
    img_src = img_src.astype(np.float32) / 255

    g_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    g_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
    i_x = cv.filter2D(img_src, -1, g_x)
    i_y = cv.filter2D(img_src, -1, g_y)
    img_dst = cv.magnitude(i_x, i_y)

    img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def filter_laplace(img_name):

    img_src = cv.imread(img_name)
    img_src = img_src.astype(np.float32) / 255

    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    img_dst = cv.filter2D(img_src, -1, kernel)

    img_dst = (255 * img_dst).clip(0, 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst)

    plt.show()

    return img_dst


def extract_canny_edges(img_name):

    img_src = cv.imread(img_name)
    img_dst = cv.Canny(img_src, 100, 200)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img_src, cmap='gray')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img_dst, cmap='gray')

    plt.show()

    return img_dst


def get_filtered(img_snp_name,
                 img_additive_name,
                 img_speckle_name,
                 img_gaussian_name,
                 img_poisson_name):

    img_snp = cv.imread(img_snp_name)
    img_additive = cv.imread(img_additive_name)
    img_speckle = cv.imread(img_speckle_name)
    img_gaussian = cv.imread(img_gaussian_name)
    img_poisson = cv.imread(img_poisson_name)

    kernel_size = (9, 9)
    use_median = True
    use_weighted = False
    img_1 = filter_weighted_rank(img_snp_name, kernel_size=kernel_size, use_median=use_median, use_weighted=use_weighted)
    img_2 = filter_weighted_rank(img_additive_name, kernel_size=kernel_size, use_median=use_median, use_weighted=use_weighted)
    img_3 = filter_weighted_rank(img_speckle_name, kernel_size=kernel_size, use_median=use_median, use_weighted=use_weighted)
    img_4 = filter_weighted_rank(img_gaussian_name, kernel_size=kernel_size, use_median=use_median, use_weighted=use_weighted)
    img_5 = filter_weighted_rank(img_poisson_name, kernel_size=kernel_size, use_median=use_median, use_weighted=use_weighted)

    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(10)
    ax = fig.add_subplot(5, 2, 1)
    plt.imshow(img_snp)
    ax.set_title("Salt and Pepper")

    ax = fig.add_subplot(5, 2, 2)
    plt.imshow(img_1)
    ax.set_title("Filtered")

    ax = fig.add_subplot(5, 2, 3)
    plt.imshow(img_additive)
    ax.set_title("Additive")

    ax = fig.add_subplot(5, 2, 4)
    plt.imshow(img_2)
    ax.set_title("Filtered")

    ax = fig.add_subplot(5, 2, 5)
    plt.imshow(img_speckle)
    ax.set_title("Speckle")

    ax = fig.add_subplot(5, 2, 6)
    plt.imshow(img_3)
    ax.set_title("Filtered")

    ax = fig.add_subplot(5, 2, 7)
    plt.imshow(img_gaussian)
    ax.set_title("Gaussian")

    ax = fig.add_subplot(5, 2, 8)
    plt.imshow(img_4)
    ax.set_title("Filtered")

    ax = fig.add_subplot(5, 2, 9)
    plt.imshow(img_poisson)
    ax.set_title("Poisson")

    ax = fig.add_subplot(5, 2, 10)
    plt.imshow(img_5)
    ax.set_title("Filtered")

    plt.savefig("comparison_filter_9_2.png", bbox_inches='tight')


def get_edges(img_name):

    img_src = cv.imread(img_name)

    edges_roberts = filter_roberts(img_name)
    edges_prewitt = filter_prewitt(img_name)
    edges_sobel = filter_sobel(img_name)
    edges_laplace = filter_laplace(img_name)
    edges_canny = extract_canny_edges(img_name)

    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(10)
    ax = fig.add_subplot(3, 2, 1)
    plt.imshow(img_src, cmap='gray')
    ax.set_title("Source Image")

    ax = fig.add_subplot(3, 2, 2)
    plt.imshow(edges_roberts, cmap='gray')
    ax.set_title("Roberts")

    ax = fig.add_subplot(3, 2, 3)
    plt.imshow(edges_prewitt, cmap='gray')
    ax.set_title("Prewitt")

    ax = fig.add_subplot(3, 2, 4)
    plt.imshow(edges_sobel, cmap='gray')
    ax.set_title("Sobel")

    ax = fig.add_subplot(3, 2, 5)
    plt.imshow(edges_laplace, cmap='gray')
    ax.set_title("Laplace")

    ax = fig.add_subplot(3, 2, 6)
    plt.imshow(edges_canny, cmap='gray')
    ax.set_title("Canny")

    plt.savefig("comparison_filter_10.png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_image')

    args = parser.parse_args()

    get_edges(args.path_to_image)
