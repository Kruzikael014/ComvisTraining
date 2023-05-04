import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import os

train_path = 'images/train'
train_dir_list = os.listdir(train_path)

# print(train_dir_list)
image_list = []
image_class_list = []

for idx, train_dir in enumerate(train_dir_list):
    dir_path = os.listdir(f'{train_path}/{train_dir}')
    for image_path in dir_path:
        image_list.append(f'{train_path}/{train_dir}/{image_path}')
        image_class_list.append(idx)

# for image_list1 in image_list:
#     print(image_list1)

sift = cv2.SIFT_create()

descriptor_list = []

for image_path in image_list:
    _, ds = sift.detectAndCompute(cv2.imread(image_path), None)
    descriptor_list.append(ds)

# yang index 0 tumpuk kedalam variabel stack ds
ds_stack = descriptor_list[0]

for ds in descriptor_list[1:]:
    ds_stack = np.vstack((ds_stack, ds))
ds_stack = np.float32(ds_stack)

# K-Means clustering
centroids, _ = kmeans(ds_stack, 100, 1)
image_features = np.zeros((len(image_list), len(centroids)), 'float32')

for i in range(0, len(image_list)):
    words, _ = vq(descriptor_list[i], centroids)
    for w in words:
        image_features[i][w] += 1

stdScaler = StandardScaler().fit(image_features)
image_features = stdScaler.transform(image_features)

svc = LinearSVC()
svc.fit(image_features, np.array(image_class_list))

# Test
test_path = 'images/test'
image_list = []

for path in os.listdir(test_path):
    image_list.append(f'{test_path}/{path}')

descriptor_list = []

for image_path in image_list:
    _, ds = sift.detectAndCompute(cv2.imread(image_path), None)
    descriptor_list.append(ds)

# yang index 0 tumpuk kedalam variabel stack ds
ds_stack = descriptor_list[0]

for ds in descriptor_list[1:]:
    ds_stack = np.vstack((ds_stack, ds))
ds_stack = np.float32(ds_stack)

# K-Means clustering
centroids, _ = kmeans(ds_stack, 100, 1)
test_features = np.zeros((len(image_list), len(centroids)), 'float32')

for i in range(0, len(image_list)):
    words, _ = vq(descriptor_list[i], centroids)
    for w in words:
        test_features[i][w] += 1

test_features = stdScaler.transform(test_features)
result = svc.predict(test_features)

for class_id, image_path in zip(result, image_list):
    print(f'{image_path} : {train_dir_list[class_id]}')