import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)

colors = ['c', 'm', 'y', 'l']
markers = ['.', '^', '*', 's']

for i in range(k):
    dataX = X[kmeans.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1], c=colors[i], marker=markers[i])
    print(dataX.size)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200, c='#0599FF')
plt.show()

