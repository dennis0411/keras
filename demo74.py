import tensorflow as tf
import pandas as pd
import keras
from sklearn.preprocessing import LabelBinarizer
from keras import callbacks
from keras import layers, models

csv1 = pd.read_csv("data/bmi.csv")
csv1['height'] = csv1['height'] / 200
csv1['weight'] = csv1['weight'] / 100
print(csv1[:10])
encoder = LabelBinarizer()
transformedLabel = encoder.fit_transform(csv1['label'])
print(csv1['label'][:10])
print(transformedLabel[:10])

test_csv = csv1[25000:]
test_pat = test_csv[['weight', 'height']]
test_ans = transformedLabel[25000:]

train_csv = csv1[:25000]
train_pat = train_csv[['weight', 'height']]
train_ans = transformedLabel[:25000]

model = models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(2,)))
model.add(layers.Dense(3, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
board = callbacks.TensorBoard(log_dir='logs/demo74')
model.fit(train_pat, train_ans, batch_size=50, epochs=100, verbose=1,
          validation_data=(test_pat, test_ans), callbacks=[board])
score = model.evaluate(test_pat, test_ans, verbose=0)