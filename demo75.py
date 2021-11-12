import tensorflow as tf

imageSourceArray = tf.constant([1, 1, 1, 0, 0, 0] * 6, tf.float32)

print(imageSourceArray)
images = tf.reshape(imageSourceArray, [1, 6, 6, 1])
images = tf.transpose(images, perm=[0, 2, 1, 3])
print(images[0, :, :, 0])
# filterSourceArray = tf.constant([1, 0, -1] * 3, tf.float32)
filterSourceArray = tf.constant([-1, 0, 1] * 3, tf.float32)
filter = tf.reshape(filterSourceArray, [3, 3, 1, 1])
filter = tf.transpose(filter, perm=[1, 0, 2, 3])
print(filter[:, :, 0, 0])
conv = tf.nn.conv2d(images, filter, [1, 1, 1, 1], padding='VALID')
convResult = conv.numpy()
print(convResult.shape)
print(convResult[0, :, :, 0])

