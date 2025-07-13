import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def load_images_and_labels(data_dir, img_size=(128, 128)):
    X, y = [], []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        label = 1 if 'blur' in class_name.lower() else 0
        for file in os.listdir(class_path):
            filepath = os.path.join(class_path, file)
            img = cv2.imread(filepath)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            X.append(img_to_array(img) / 255.0)
            y.append(label)
    return np.array(X), np.array(y)

def preprocess_single_image(image, img_size=(128, 128)):
    image = image.resize(img_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)
