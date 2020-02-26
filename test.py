import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import os

from matplotlib import cm

from tensorflow.keras.datasets import fashion_mnist
(_, _), (x_test, y_test) = fashion_mnist.load_data()

# load Model
fashion_model = tf.keras.models.load_model('fashion_model.h5')

# The first two variables are x_train and x_test, we don't use it here.

fashion_mnist_labels = np.array([
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'])


'''Convert pixels into floating point number,
and reshape it into (1,28,28,1) which is
(number_of_sample, width, height, channel).
'''
def convertMnistData(image):
    img = image.astype('float32')
    img /= 255

    return image.reshape(1,28,28,1)


plt.figure(figsize=(16,16))

right = 0
mistake = 0
predictionNum = 100

for i in range(predictionNum):
    # Random one image from a test set.
    index = random.randint(0, x_test.shape[0])
    image = x_test[index]
    data = convertMnistData(image)
    
    plt.subplot(10, 10, i+1)
    plt.imshow(image,  cmap=cm.gray_r)
    plt.axis('off')
    plt.gca().grid(False)
    
    prediction = fashion_model.predict(data, batch_size=1)

    best_prediction = 0.0
    best_class = 0
    # Loop through all classes to get the class that has highest prediction value.
    for label in [0,1,2,3,4,5,6,7,8,9]:
        if best_prediction < prediction[0][label]:
            best_prediction = prediction[0][label]
            best_class = label
            # print(best_class)

    # If the model prediction is correct display this text:
    # Else display predicted class and labelled class.
    if y_test[index] == best_class:
        plt.title(fashion_mnist_labels[best_class])
        right += 1
    else:
        plt.title(fashion_mnist_labels[best_class] + "!=" + fashion_mnist_labels[y_test[index]], color='#ff0000')
        mistake += 1

plt.show()
print("The number of correct answers:", right)
print("The number of mistake:", mistake)
print("A correct answer rate:", right / (mistake + right) * 100, '%')
