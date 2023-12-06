# coding: utf-8
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=(None,), name='x')
a = tf.Variable(2.0, name='a')
b = tf.Variable(3.0, name='b')
c = tf.Variable(3.0, name='c')

# d = tf.stop_gradient(a) + b
d = a + b
e = a + c

f = d + e

grads = tf.gradients(f, [a])

# 执行计算
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 假设有一个输入数据 x_input
    x_input = [1.0, 2.0, 3.0]

    # 计算 c 的值
    f_output, g = sess.run([f, grads], feed_dict={x: x_input})
    print("f:", f_output)
    print("g:", g)
