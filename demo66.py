import numpy as np
from keras import layers, models
from keras.datasets import imdb
from keras.callbacks import TensorBoard

MAX_WORDS = 15000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=MAX_WORDS)
print(train_data[0])
print(max([max(sequence) for sequence in train_data]))

word_index = imdb.get_word_index()
print(type(word_index))
print(list(word_index.items())[:10])

reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])
# print(reverse_word_index[0])
print(reverse_word_index[1])
print(reverse_word_index[2])
print(reverse_word_index[3])
print(reverse_word_index[4])
for i in range(5):
    decoded_review = ' '.join([reverse_word_index.get(i - 3, "?") for i in train_data[i]])
    print(decoded_review)


def vectorize_sequence(sequences, dimension=MAX_WORDS):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


X_train = vectorize_sequence(train_data)
X_test = vectorize_sequence(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(X_train[0])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(MAX_WORDS,)))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
callback1 = TensorBoard(log_dir="logs", histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train, y_train, epochs=30, batch_size=256,
          validation_data=(X_test, y_test), callbacks=[callback1])


