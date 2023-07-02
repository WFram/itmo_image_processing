import cv2 as cv
import matplotlib.pyplot as plt


def calc_hist_cv(img):

    # Number of histogram bins
    hist_size = 256

    # Histogram range
    hist_range = (0, 256)

    # Calculate a histogram for each layer
    hist = []
    for i in range(3):
        hist.append(cv.calcHist(img, [i], None, [hist_size], hist_range))

    return hist


def calc_hists_plt(img, img_new, hist_name):

    # Number of histogram bins
    hist_size = 256

    # Histogram range
    hist_range = [0, 256]

    # Split an image into color layers
    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]

    img_new_b = img_new[:, :, 0]
    img_new_g = img_new[:, :, 1]
    img_new_r = img_new[:, :, 2]

    # Calculate and plot histograms
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
