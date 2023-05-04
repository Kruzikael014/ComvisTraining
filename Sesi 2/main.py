import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('model.jpg')
grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_result(label = None, image = None, c_map = 'gray'):
    # Plot kiri atas
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.hist(image.flat, bins=256, range=(0, 256))
    plt.title(label)
    plt.xlabel('Intensity Value')
    plt.ylabel('Intensity Quantity')

    # Plot kanan atas
    plt.subplot(1, 2, 2)
    plt.imshow(image, c_map)
    plt.axis('off')
    plt.show()

normal_image = grayed_image.copy()
show_result('normal image', normal_image)

# Normal Equalization (pakai bawain function)
equalized_hist = cv2.equalizeHist(grayed_image)
show_result('normalized image', equalized_hist)

# CLAHE (Contrast Limiting Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahed_hist = clahe.apply(grayed_image)

show_result('normalized image (CLAHE)', clahed_hist)

hist_labels = ['normal', 'normalized', 'normalized (clahe)']
hist_images = [normal_image, equalized_hist, clahed_hist]

plt.figure(figsize=(12, 12))
for i, (lbl, img) in enumerate(zip(hist_labels, hist_images)):
    plt.subplot(3,1,i+1)
    plt.hist(img.flat, bins=256, range=(0, 256))
    plt.title(lbl)
    plt.xlabel('Intensity Value')
    plt.ylabel('Intensity Quantity')
plt.show()


plt.figure(figsize=(12, 12))
for i, (lbl, img) in enumerate(zip(hist_labels, hist_images)):
    plt.subplot(1,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(lbl)
    plt.axis('off')
plt.show()