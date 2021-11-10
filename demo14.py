import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a
c = a.view()
d = a.copy()
print(a, b, c, d, sep='\n')
a[0][0] = 100
print("after change a[0][0] to 100")
print(a, b, c, d, sep='\n')
a.shape = (4,)
print("after change a shape = (4,)")
print(a, b, c, d, sep='\n')