import cv2
import numpy as np
from utils import *
from keras.models import load_model
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tkinter import *
from PIL import ImageGrab
from PIL import Image

batch_size = 128
num_classes = 10
epochs = 30
input_shape = (28, 28, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model = load_model('mnist.h5')
print("geladebn")
def defineBiggestContour(contours):
    biggest_contour = []
    biggest_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
            if area > biggest_area and len(approx) == 4:
                biggest_contour = approx
                biggest_area = area
    return biggest_contour, biggest_area


def orderPoints(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def predict_digit(img):
    print(img)
    img = cv2.resize(img, (28,28))
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.0
    img = 1 - img
    res = model.predict([img])[0]
    return np.argmax(res), max(res)
