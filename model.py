from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

def build_simple_cnn(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape, name="conv2d"),
        MaxPooling2D(name="max_pooling2d"),
        Conv2D(64, (3,3), activation='relu', name="conv2d_1"),
        MaxPooling2D(name="max_pooling2d_1"),
        Flatten(name="flatten"),
        Dense(64, activation='relu', name="dense"),
        Dense(1, activation='sigmoid', name="dense_1")
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

'''
def build_mobilenetv2(input_shape=(128, 128, 3)):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
'''