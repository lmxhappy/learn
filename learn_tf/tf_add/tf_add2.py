#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/4/9
'''
import tensorflow as tf

a = tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)
b = tf.constant([1, -1], dtype=tf.float32)
c = tf.constant([1], dtype=tf.float32)

with tf.Session() as sess:
    print('a:')
    a = sess.run(a)
    print(a)

    print('b:')
    b = sess.run(b)
    print(b)

    print('bias_add:')
    print(sess.run(tf.nn.bias_add(a, b)))
    # 执行下面语句错误
    # print(sess.run(tf.nn.bias_add(a, c)))