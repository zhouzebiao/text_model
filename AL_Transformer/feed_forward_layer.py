# -*- coding: utf-8 -*-
"""
 Created by zaber on 2020-01-03 14:
"""
import numpy as np

import tensorflow as tf


def gelu(x):
    """Gaussian Error Linear Unit.
    Original paper: https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_size, filter_size, gelu_dropout, train):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        # self.intermediate_size = intermediate_size
        self.gelu_dropout =gelu_dropout
        self.train = train
        self.filter_dense_layer = None
        self.output_dense_layer = None

    def build(self, input_shape):
        self.filter_dense_layer = tf.compat.v1.layers.Dense(self.filter_size, use_bias=True,
                                                            activation=tf.nn.leaky_relu,
                                                            name='filter_layer', dtype=self.dtype)
        self.output_dense_layer = tf.compat.v1.layers.Dense(self.hidden_size, use_bias=True, name='output_layer',
                                                            dtype=self.dtype)

    def call(self, inputs, **kwargs):
        output = self.filter_dense_layer(inputs)
        if self.train:
            output = tf.nn.dropout(output, rate=self.gelu_dropout)
        output = self.output_dense_layer(output)

        return output
