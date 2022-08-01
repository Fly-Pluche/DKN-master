# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 13:29:59 2022

@author: 磷Sandwich
"""


import tensorflow as tf

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess,"../model/DKNModel")
    
