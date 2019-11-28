# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-11-26 17:
"""

import attention_layer
import tensorflow as tf


class AttentionLayersTest(tf.test.TestCase):

    def test_attention_layer(self):
        hidden_size = 64
        num_heads = 4
        dropout = 0.5
        layer = attention_layer.SelfAttention(hidden_size, num_heads, dropout)
        length = 2
        x = tf.ones([1, length, hidden_size])
        bias = tf.ones([1])
        cache = {
            "k": tf.zeros([1, 0, hidden_size]),
            "v": tf.zeros([1, 0, hidden_size]),
        }
        y = layer(x, bias, training=True, cache=cache)
        self.assertEqual(y.shape, (1, length, 64,))
        self.assertEqual(cache["k"].shape, (1, length, 64,))
        self.assertEqual(cache["v"].shape, (1, length, 64,))


if __name__ == "__main__":
    tf.test.main()
"""
export PYTHONPATH=/data/model;CUDA_VISIBLE_DEVICES='7' python attention_layer_test.py
"""
