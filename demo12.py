import numpy as np

a = np.zeros((10, 2))
print(a)
b = a.T
c = b.view()
print(a.shape, b.shape, c.shape)
d = np.reshape(b, (5, 4))
print(d)
print(a.shape, b.shape, c.shape, d.shape)
e = np.reshape(b, (20,))
print(e.shape)
f = np.reshape(b, (-1,))
print(f.shape)
g = np.reshape(b, (20, -1))
print(g.shape)
h = np.reshape(b, (-1, 20))
print(h.shape)