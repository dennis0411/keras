import tensorflow as tf

x = tf.Variable(tf.random.uniform((2, 2)))
print(x)

with tf.GradientTape() as tape:
    y = 5 * x ** 2 + 6 * x + 4
    diff_1 = tape.gradient(y, x)
print("x=", x.numpy(), sep="\n")
# 10*x+6
# 10*0.8370+6
print("diff2=", diff_1.numpy(), sep="\n")

W = tf.Variable(tf.random.uniform((1, 1)))
b = tf.Variable(tf.zeros((1,)))
#x = tf.random.uniform((1, 1))
x = tf.Variable(tf.random.uniform((1, 1)))
with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + 2 * b
    grad1 = tape.gradient(y, [W, b, x])
print("x=", x.numpy())
print("w=", W.numpy())
print("b=", b.numpy())
print("y=", y.numpy())
print(grad1[0].numpy())
print(grad1[1].numpy())
print(grad1[2].numpy())