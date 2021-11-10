from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 0], [0, 1], [1, 2], [1, 4], [1, 6],
              [-1, 0], [4, 2], [4, 4], [4, 0], [5, 6], [5, 7]])
kmeans = KMeans(n_clusters=2).fit(X)

print("labels=", kmeans.labels_)
print("centers=", kmeans.cluster_centers_)

newX = [[2, 2], [0, 0], [4, 4], [6, 6], [8, 8]]
print("predict as", kmeans.predict(newX))
print("kmean inertia=", kmeans.inertia_)