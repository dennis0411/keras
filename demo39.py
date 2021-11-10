import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
classifier = GaussianNB()
classifier.fit(X, Y)
newX = [[1, 0], [0, 1], [-1, 0], [0, -1]]
print(classifier.predict(newX))

classifier2 = GaussianNB()
classifier2.partial_fit(X, Y, np.unique(Y))
print(classifier2.predict(newX))
classifier2.partial_fit([[0, 0]], [1])
print(classifier2.predict(newX))