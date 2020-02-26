import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import os

from matplotlib import cm
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import fashion_mnist
import logging
tf.get_logger().setLevel(logging.ERROR)
img_rows, img_cols = 28, 28
num_classes = 10


def prep_data(raw):
    # Get all labels from the first columns for all rows.
    y = raw[:, 0]
    # Convert labels to one-hot encoding format
    out_y = keras.utils.to_categorical(y, num_classes)
    # Get all 784 pixels for all rows.
    x = raw[:, 1:]
    # Get number of rows from CSV (Number of images).
    num_images = raw.shape[0]
    # Reshape all pixels into 60000 of images with
    # 28x28 pixels, 1 channel.
    # 1 channel = greyscale
    # (60000, 28, 28, 1)
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    # Divide all pixels by 255 to get floating point values.
    out_x = out_x / 255

    return out_x, out_y


fashion_train_file = "./fashionmnist/fashion-mnist_train.csv"
# Load CSV file into NumPy array, skip the first line.
fashion_train_data = np.loadtxt(fashion_train_file, skiprows=1, delimiter=',')


x, y = prep_data(fashion_train_data)

gray = np.squeeze(x[1])
plt.figure()
plt.imshow(gray)
plt.colorbar()
plt.gca().grid(False)
plt.show()

# The Sequential model is a linear stack of layers.
fashion_model = Sequential()

fashion_model.add(Conv2D(
    filters=12,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(img_rows, img_cols, 1)) #28x28x1
)

fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu')) #1
fashion_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu')) #2
fashion_model.add(Flatten()) # เรียงเป็นแนวดิ่ง
fashion_model.add(Dense(100, activation='relu'))
fashion_model.add(Dense(num_classes, activation='softmax'))
fashion_model.summary()

sgd = optimizers.SGD(
    lr=0.01,
    decay=0,
    momentum=0
)

fashion_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=sgd, #stochastic gradient descent
    metrics=['accuracy']
)

# Check TensorFlow version and GPU device
if K.backend() == "tensorflow":
    device_name = tf.test.gpu_device_name()
    if device_name == '':
        device_name = "None"
    print('Using TensorFlow version:', tf.__version__, ', GPU:', device_name)

'''Batch size 32 will update weights every 32 training samples.
validation_split into 48,000 samples for training,
12,000 for validation.
'''
fashion_model.fit( #fit with keras
    x,
    y,
    batch_size=32,
    epochs=35,
    validation_split=0.2
)

fashion_model.save('fashion_model.h5')

# #validate ไม่ควรสูงกว่า train!!!

