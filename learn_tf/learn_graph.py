#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/11/13
'''

import tensorflow as tf
# a = tf.constant([1,2,3], name='a')
# print(a.graph)
# print(a.graph is tf.get_default_graph())

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable('v', shape=[1], initializer=tf.ones_initializer)

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable('v', shape=[1], initializer=tf.zeros_initializer)

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):

        # 从图里取出那个张量
        the_v = tf.get_variable('v')
        # print(the_v)
        # print(sess.run(the_v))
        print('-------')
        print(tf.GraphKeys.GLOBAL_VARIABLES)
        print(tf.GraphKeys.LOCAL_VARIABLES)


# with tf.Session(graph=g2) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable('v')))