import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1), padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1), padding='valid'))
model.add(layers.MaxPooling2D((2, 2)))
# extract from feature map
model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, padding='valid'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, padding='valid'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation=tf.nn.relu))
model.add(layers.Dense(10, activation=tf.nn.softmax))
model.summary()


