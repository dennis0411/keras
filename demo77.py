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
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=32)

