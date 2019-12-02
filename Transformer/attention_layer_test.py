# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-11-26 17:
"""

import attention_layer
import tensorflow as tf


class AttentionLayersTest(tf.test.TestCase):

    def test_attention_layer(self):
        embedding_size = 64
        hidden_size = 512
        num_heads = 4
        dropout = 0.5
        layer = attention_layer.SelfAttention(embedding_size, hidden_size, num_heads, dropout)
        length = 2
        x = tf.ones([1, length, embedding_size])
        bias = tf.ones([1])
        cache = {
            "k": tf.zeros([1, 0, embedding_size]),
            "v": tf.zeros([1, 0, embedding_size]),
        }
        inputs = {'query_input': x, 'source_input': x, 'bias': bias, 'training': True, 'cache': cache}
        y, cache = layer(inputs)
        # y = layer(x,bias,True,cache)
        self.assertEqual(y.shape, (1, length, 512,))
        self.assertEqual(cache["k"].shape, (1, length, embedding_size,))
        self.assertEqual(cache["v"].shape, (1, length, embedding_size,))


if __name__ == "__main__":
    tf.test.main()
"""
export PYTHONPATH=/data/model;CUDA_VISIBLE_DEVICES='7' python attention_layer_test.py
1048576
45056
"""
