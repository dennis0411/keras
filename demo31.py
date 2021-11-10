import time

from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from numpy import mean
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
data = iris.data
target = iris.target

regression1 = LogisticRegression(max_iter=50000)
svc1 = SVC(kernel='linear', C=100)
svc2 = SVC(kernel='poly', C=100)
svc3 = SVC(kernel='rbf', C=100)
tree1 = DecisionTreeClassifier()

classifiers = [regression1, svc1, svc2, svc3, tree1]
for c in classifiers:
    score = model_selection.cross_val_score(c, data, target, cv=3)
    print(c, score, mean(score))