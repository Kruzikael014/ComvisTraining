import cv2
import os 
import numpy as np
import math

CLASSIFIER = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Train
TRAIN_PATH = 'images/train'
TRAIN_DIR  = os.listdir(TRAIN_PATH)
face_list  = []
class_list = []

for idx, train_dir in enumerate(TRAIN_DIR):
    for image_path in os.listdir(f'{TRAIN_PATH}/{train_dir}'):
        path = f'{TRAIN_PATH}/{train_dir}/{image_path}'
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        faces = CLASSIFIER.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
        
        if len(faces) < 1:
            continue
        else:
            for face_rect in faces:
                x, y, w, h = face_rect
                face_image = gray[y: y + w, x : x + h]
                face_list.append(face_image)
                class_list.append(idx)
        # print(path)

FACE_RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()
FACE_RECOGNIZER.train(face_list, np.array(class_list))

# Test
TEST_PATH = 'images/test'

for path in os.listdir(TEST_PATH):
    full_path = f'{TEST_PATH}/{path}'
    image = cv2.imread(full_path)
    grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = CLASSIFIER.detectMultiScale(grayed_image, scaleFactor = 1.2, minNeighbors = 5)

    if len(faces) < 1:
        continue
    else:
        for face_rect in faces:
            x, y, w, h = face_rect
            face_image = grayed_image[y: y + w, x : x + h]

            res, conf = FACE_RECOGNIZER.predict(face_image)

            conf = math.floor(conf * 100) / 100
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            image_text = f'{TRAIN_DIR[res]} : {conf}%'
            cv2.putText(image, image_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 1)
            cv2.imshow('result', image)
            cv2.waitKey()
            cv2.destroyAllWindows()

cv2.destroyAllWindows()