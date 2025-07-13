import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocess import load_images_and_labels
from model import build_simple_cnn
# from model import build_mobilenetv2

DATA_DIR = 'data/'
MODEL_PATH = 'saved_models/best_model.h5'
IMG_SIZE = (128, 128)

X, y = load_images_and_labels(DATA_DIR, img_size=IMG_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = build_simple_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, callbacks=[checkpoint])
