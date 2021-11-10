from numpy import array
from sklearn.decomposition import PCA

A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)

pca1 = PCA(2)
pca1.fit(A)
print(pca1)
print(pca1.components_)
print(pca1.explained_variance_)
print(pca1.explained_variance_ratio_)
B = pca1.transform(A)
print(B)