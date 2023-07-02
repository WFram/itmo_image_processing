import numpy as np
import skimage
import cv2 as cv

def apply_salt_n_pepper_noise(img_name, amount=0.4):
    img = cv.imread(img_name)

    img_noise = skimage.util.random_noise(img, 's&p', amount=amount)
    img_noise = (255 * img_noise).clip(0, 255).astype(np.uint8)

    return img_noise


def apply_additive_noise(img_name, mean=0, var=0.4):
    img = cv.imread(img_name)

    img_noise = skimage.util.random_noise(img, 'additive', mean=mean, var=var)
    img_noise = (255 * img_noise).clip(0, 255).astype(np.uint8)

    return img_noise


def apply_speckle_noise(img_name, mean=2, var=0.2):
    img = cv.imread(img_name)

    img_noise = skimage.util.random_noise(img, 'speckle', mean=mean, var=var)
    img_noise = (255 * img_noise).clip(0, 255).astype(np.uint8)

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

    return img_noise


def apply_poisson_noise(img_name, peak=5.0):
    img = cv.imread(img_name)
    img_new = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img

    noise_mask = np.random.poisson(img_new * peak)
    img_noise = img_new + noise_mask
    if img.dtype == np.uint8:
        img_noise = (255 * img_noise).clip(0, 255).astype(np.uint8)

    return img_noise
