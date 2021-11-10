import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
species = iris.target

X_reduced = PCA(n_components=3).fit_transform(iris.data)
print(X.shape, X_reduced.shape)

fig = plt.figure(1, figsize=(9, 9))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=species)
ax.set_xlabel("first eigen")
ax.set_ylabel("second eigen")
ax.set_zlabel("third eigen")
plt.show()