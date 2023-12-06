#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/11/13
'''
import tensorflow as tf
from numpy.random import RandomState
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# x不是变量
# x = tf.constant([[0.3, 0.7]])
x = tf.placeholder(tf.float32, shape=(None,2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1), name='y-input')

# 为啥a、y不是变量呢？？？
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y_head = tf.sigmoid(y)

cross_entropy = - tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y_head, 1e-10, 1.0))
    + (1-y_)*tf.log(tf.clip_by_value(1-y_head, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2<1)] for x1, x2 in X]

batch_size = 8
# tf.assign(w1, w2)
with tf.Session() as sess:
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    # print(sess.run(y, feed_dict={x:[[0.3, 0.7]]}))
    # # print(tf.global_variables())
    # print(tf.trainable_variables())
    # # name_list = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # # print(name_list)

    # 初始化variable_w1,w2
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})

        if i % 1000 == 0:
            # 在所有数据上算一个值
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print(f"After {i} training step(s), cross entropy on all data is {total_cross_entropy}")

    print(sess.run(w1))
    print(sess.run(w2))