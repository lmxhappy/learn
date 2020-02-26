

import tensorflow as tf

a = tf.constant([[1,2], [3,4], [4,6]], shape=[3, 2])
b = tf.constant([[1], [3], [4]], shape=[3, 1])

c = tf.add(a, b)

# broadcast
with tf.Session() as sess:
    print(sess.run(c))
