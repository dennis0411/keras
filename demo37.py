import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

URL1 = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
df1 = pd.read_csv(URL1, header=None, prefix='X')
print(df1.shape)
data, labels = df1.iloc[:, :-1], df1.iloc[:, -1]
print(data.shape)
print(labels.shape)
print(np.unique(labels, return_counts=True))
df1.rename(columns={'X60': 'Label'}, inplace=True)
print(df1.columns)

classifier = KNeighborsClassifier(n_neighbors=4)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
print("score=", classifier.score(X_test, y_test))

result_cm1 = confusion_matrix(y_test, y_predict)
print(result_cm1)