import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_result(source, cmap = None):
    plt.imshow(source, cmap)
    plt.show()

image = cv2.imread("./chessboard.jpg")
grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayed_image = np.float32(grayed_image)

harris = cv2.cornerHarris(grayed_image, 2, 5, 0.04)

without_subpix = image.copy()
without_subpix[harris > 0.01 * harris.max()] = [0, 255, 0]

show_result(harris, 'gray')
show_result(without_subpix, 'gray')

_, thresh = cv2.threshold(harris, 0.01 * harris.max(), 255, 0)
thresh = np.uint8(thresh)

_, _, _, centroids = cv2.connectedComponentsWithStats(thresh)
centroids = np.float32(centroids)

criteria = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 100, 0.0001)

enhanced_criteria = cv2.cornerSubPix(grayed_image, centroids, (2, 2), (-1, -1), criteria)

enhanced_criteria = np.uint16(enhanced_criteria)

with_subpix = image.copy()

for i in enhanced_criteria:
    x, y = i[:2]
    with_subpix[y, x] = [255, 0, 0]

show_result(with_subpix, 'gray')