#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/11/15
'''

import tensorflow as tf
v = tf.constant([1.0, 2.0, 3.0])
v2 = tf.constant([1.0, 2.0, 3.0])
v3 = tf.constant([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(tf.log(v).eval())

    # print((v*v2).eval())
    print(tf.reduce_mean(v3).eval())
    print(tf.reduce_mean(v2).eval())