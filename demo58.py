import numpy as np
from keras.models import Sequential
from keras.layers import Dense

FILENAME = 'data/diabetes.csv'
dataset1 = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
print(type(dataset1))
print(dataset1.shape)
inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)

model = Sequential()
model.add(Dense(14, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(inputList, resultList, epochs=200, batch_size=20)
scores = model.evaluate(inputList, resultList)
print(type(scores))
print(scores)
print(model.metrics_names)
for s, m in zip(scores, model.metrics_names):
    print("{}={}".format(m, s))