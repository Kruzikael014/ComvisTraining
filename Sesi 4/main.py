import cv2
import matplotlib.pyplot as plt
import numpy as np

KSIZE = 3

image = cv2.imread('fruits.jpg')
grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_result(n_row=None, n_col=None, res_stack=None):
    plt.figure(figsize=(12, 12))
    for i, (lbl, img) in enumerate(res_stack):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(lbl)
        plt.axis('off')
    plt.show()


# - Laplacian (CV_8U, CV_16S, CV_32F, CV_64F)
laplaced_8u = cv2.Laplacian(grayed_image, cv2.CV_8U)
laplaced_16s = cv2.Laplacian(grayed_image, cv2.CV_16S)
laplaced_32f = cv2.Laplacian(grayed_image, cv2.CV_32F)
laplaced_64f = cv2.Laplacian(grayed_image, cv2.CV_64F)

laplace_labels = ['8u', '16s', '32f', '64f']
laplace_images = [laplaced_8u, laplaced_16s, laplaced_32f, laplaced_64f]

show_result(2, 2, zip(laplace_labels, laplace_images))

sobel_x = cv2.Sobel(grayed_image, cv2.CV_32F, 1, 0, KSIZE)
sobel_y = cv2.Sobel(grayed_image, cv2.CV_32F, 0, 1, KSIZE)
merged_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
merged_sobel *= 255/merged_sobel.max()

sobel_labels = ['sobel_x', 'sobel_y', 'merged_sobel']
sobel_images = [sobel_x, sobel_y, merged_sobel]

show_result(1, 3, zip(sobel_labels, sobel_images))

canny_50100 = cv2.Canny(grayed_image, 50, 100)
canny_50150 = cv2.Canny(grayed_image, 50, 150)
canny_75150 = cv2.Canny(grayed_image, 75, 150)
canny_75225 = cv2.Canny(grayed_image, 75, 225)

canny_labels = ['50100', '50150', '75150', '75225']
canny_images = [canny_50100, canny_50150, canny_75150, canny_75225]

show_result(2, 2, zip(canny_labels, canny_images))
