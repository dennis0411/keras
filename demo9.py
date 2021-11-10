import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import numpy

regressionData = datasets.make_regression(100, 1, noise=5)
print(type(regressionData))
print(regressionData[0].shape, regressionData[1].shape)
print(type(regressionData[0]))
plt.scatter(regressionData[0], regressionData[1], c='red', marker='^')
plt.show()

regression1 = linear_model.LinearRegression()
regression1.fit(regressionData[0], regressionData[1])

print(f"coef={regression1.coef_[0]}, intercept={regression1.intercept_}")
print(f"score={regression1.score(regressionData[0], regressionData[1])}")

range1 = numpy.arange(regressionData[0].min() - 0.5, regressionData[0].max() + 0.5, 0.01)
plt.plot(range1, regression1.coef_*range1+regression1.intercept_)
plt.scatter(regressionData[0], regressionData[1], c='red', marker='^')
plt.show()