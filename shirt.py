import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
from PIL import Image

img=Image.open('s1.jpg').convert('L')
new_width  = 28
new_height = 28
img = img.resize((new_width, new_height), Image.ANTIALIAS)



def convertMnistData(image):
    image = np.asarray(image,dtype="int32")
    img = image.astype('float')
    img /= 255
    imgplot = plt.imshow(img)
    plt.colorbar()
    plt.gca().grid(False)
    plt.show()

    return image.reshape(1,28,28,1)

data = convertMnistData(img)
fashion_model = tf.keras.models.load_model('fashion_model.h5')

prediction = fashion_model.predict(data, batch_size=1)

best_prediction = 0.0
best_class = 0
# Loop through all classes to get the class that has highest prediction value.
for label in [0,1,2,3,4,5,6,7,8,9]:
    if best_prediction < prediction[0][label]:
        best_prediction = prediction[0][label]
        best_class = label
        print(best_class)

# If the model prediction is correct display this text:
# Else display predicted class and labelled class.
# if y_test[index] == best_class:
#     plt.title(fashion_mnist_labels[best_class])
#     right += 1
# else:
#     plt.title(fashion_mnist_labels[best_class] + "!=" + fashion_mnist_labels[y_test[index]], color='#ff0000')
#     mistake += 1