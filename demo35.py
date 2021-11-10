# K efficient

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

Q = 5000
X = np.r_[np.random.randn(Q, 2) + [3, 3],
          np.random.randn(Q, 2) + [0, -3],
          np.random.randn(Q, 2) + [-3, 3]]
inertias = []

for k in range(1, 10):
    kmeans = KMeans(n_init=5, n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
print(inertias)
plt.plot(range(1, 10), inertias)
plt.show()
