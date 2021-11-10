import matplotlib.pyplot as plt
from sklearn import datasets

regression1 = datasets.make_regression(10, 6, noise=5)

for i in range(len(regression1[0])):
    x1 = regression1[0][:, i]
    y = regression1[1]
    plt.scatter(x1, y)
    plt.title("#{} V.S. Y variable".format(i))
    plt.show()
