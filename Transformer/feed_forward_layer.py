# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-12-03 14:
"""
import tensorflow as tf


class FeedForwardNetwork(tf.layers.Layer):
    def __init__(self, hidden_size, filter_size, relu_dropout, train):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train
        self.filter_dense_layer = None
        self.output_dense_layer = None

    def build(self, input_shape):
        self.filter_dense_layer = tf.layers.Dense(self.filter_size, use_bias=True, activation=tf.nn.leaky_relu,
                                                  name='filter_layer')
        self.output_dense_layer = tf.layers.Dense(self.hidden_size, use_bias=True, name='output_layer')

    def call(self, inputs, **kwargs):
        output = self.filter_dense_layer(inputs)
        if self.train:
            output = tf.nn.dropout(output, rate=self.relu_dropout)
        output = self.output_dense_layer(output)

        return output
