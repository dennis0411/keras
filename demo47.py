import tensorflow as tf
import numpy as np

a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
print(a + b)
print(np.add(a, b))

t1 = tf.constant([5, 3, 8])
t2 = tf.constant([3, -1, 2])
print(t1 + t2)
print((t1 + t2).numpy())
print(tf.add(t1, t2))
print(np.add(t1, t2))
