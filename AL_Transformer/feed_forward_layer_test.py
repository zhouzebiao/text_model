# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-11-26 17:
"""

import feed_forward_layer
import tensorflow as tf


class FeedForwardNetworkTest(tf.test.TestCase):

    def test_attention_layer(self):
        hidden_size = 64
        filter_size = 32
        relu_dropout = 0.5
        train = True
        layer = feed_forward_layer.FeedForwardNetwork(hidden_size, filter_size, relu_dropout, train)
        length = 2
        x = tf.ones([1, length, hidden_size])
        y = layer(x)
        self.assertEqual(y.shape, (1, length, hidden_size,))


if __name__ == "__main__":
    tf.test.main()
"""
export PYTHONPATH=/data/model;CUDA_VISIBLE_DEVICES='7' python feed_forward_layer_test.py
1048576
45056
"""
