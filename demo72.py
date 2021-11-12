import tensorflow as tf
import keras
from keras import datasets
from matplotlib import pyplot as plt
from keras.layers import Dense, Flatten
import numpy

(train_images, train_labels,), (test_images, test_labels) = datasets.fashion_mnist.load_data()
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(numpy.unique(train_labels, return_counts=True))
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
print(train_images[0])
train_images = train_images / 255.0
test_images = test_images / 255.0
# OFFSET = 0
# plt.figure(figsize=(10, 8))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images[i + OFFSET], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i + OFFSET]])
# plt.show()

layers = [Flatten(input_shape=(28, 28)),
          Dense(128, activation='relu'),
          Dense(64, activation='relu'),
          Dense(10, activation='softmax')]
model = keras.Sequential(layers)
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=1)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols

import numpy as np


def plot_image(i, predictions_array, true_label, image):
    true_label = true_label[i]
    image = image[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisPlot = plt.bar(range(10), predictions_array, color='#888888')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisPlot[predicted_label].set_color('red')
    thisPlot[true_label].set_color('blue')
    pass


predictions = model.predict(test_images)
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


