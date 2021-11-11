import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

FILE_NAME = 'data/iris.data'
df1 = pd.read_csv(FILE_NAME, header=None)
dataset = df1.values
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
print(features.shape)
print(labels.shape)
print(np.unique(labels, return_counts=True))
print(df1.describe())

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(np.unique(encoded_Y, return_counts=True))
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y.shape)
print(dummy_y[:5])


def baselineModel():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam',
                  metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baselineModel, epochs=200, batch_size=10, verbose=1)
kfold = KFold(n_splits=3, shuffle=True)
result = cross_val_score(estimator, features, dummy_y, cv=kfold)
print("acc={}, std={}".format(result.mean(), result.std()))