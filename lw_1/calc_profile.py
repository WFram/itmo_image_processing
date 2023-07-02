import cv2 as cv
import matplotlib.pyplot as plt


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
