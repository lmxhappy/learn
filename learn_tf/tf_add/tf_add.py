#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/4/9
'''
import tensorflow as tf

y_bias = tf.Variable(tf.random_normal([2, ], stddev=0.35), name="y_bias" )#shape=[2]
activation = tf.Variable(tf.random_normal([2, 4], stddev=0.35), name="activation") #, shape=[2, 4]
r = tf.Variable(tf.random_normal([4, 2], stddev=0.35), name="r") #shape=[4, 2]
mul = tf.matmul(activation, r)
# print(mul.shape)
# print(y_bias.shape)
print(type(mul), type(y_bias))
y = mul + y_bias

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_bias, mul, y = sess.run([y_bias, mul, y])
    print(y_bias, mul, y)
    print(y_bias.shape, mul.shape, y.shape)
