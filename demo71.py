import tensorflow as tf
import keras
import numpy as np
from keras import layers, models
from matplotlib import pyplot as plt
import pandas as pd

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

FLATTEN_DIM = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
print(type(trainImages[0]), trainImages.shape, trainImages[0].shape)

trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
trainImages /= 255
testImages /= 255
print(trainImages[0])

NUM_DIGITS = 10
trainLabels = keras.utils.np_utils.to_categorical(train_labels, NUM_DIGITS)
testLabels = keras.utils.np_utils.to_categorical(test_labels, NUM_DIGITS)
print(trainLabels[:5])

model = models.Sequential()
model.add(layers.Dense(128, activation=tf.nn.relu, input_shape=(FLATTEN_DIM,)))
model.add(layers.Dense(10, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
board = keras.callbacks.TensorBoard(log_dir='logs/demo71/', histogram_freq=0,
                                    write_graph=True, write_images=True)
model.fit(trainImages, trainLabels, epochs=20, callbacks=[board])

predictResult = model.predict(testImages)
print(predictResult[:5])

predict = np.argmax(predictResult, axis=-1)
print(predict[:5])

loss, accuracy = model.evaluate(testImages, testLabels)
print(loss, accuracy)


def plotTestImage(index):
    plt.title("the test image is %d" % test_labels[index])
    plt.imshow(test_images[index])
    plt.show()


plotTestImage(5)

trainHistory = model.fit(trainImages, trainLabels, epochs=20, validation_split=0.1)

plt.plot(trainHistory.history['accuracy'], color='red')
plt.plot(trainHistory.history['val_accuracy'], color='green')
plt.legend(['training', 'validation'])
plt.show()

cross = pd.crosstab(test_labels, predict, rownames=['label'], colnames=['predict'])
print(cross)

measure1 = pd.DataFrame({'label': test_labels, 'predict': predict})
# depend on your result
error_2_7 = measure1[(measure1.label == 7) & (measure1.predict == 2)]
print(error_2_7)
print(error_2_7.index)
for i in error_2_7.index:
    plotTestImage(i)


