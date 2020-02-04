# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-12-03 11:
"""

import embedding_layer
import tensorflow as tf


class EmbeddingLayersTest(tf.test.TestCase):

    def test_embedding_layer(self):
        vocab_size = 50
        batch_size = 32
        hidden_size = 256
        embedding_size = 64
        length = 2
        layer = embedding_layer.EmbeddingSharedWeights(vocab_size, embedding_size, hidden_size, 0.02)

        inputs = tf.ones([batch_size, length], dtype="int32")
        y = layer(inputs)
        self.assertEqual(y.shape, (batch_size, length, hidden_size,))
        x = tf.ones([1, length, embedding_size])
        output = layer.linear(x)
        self.assertEqual(output.shape, (1, length, vocab_size,))


if __name__ == "__main__":
    tf.test.main()
"""
export PYTHONPATH=/opt/model;CUDA_VISIBLE_DEVICES='1' python embedding_layer_test.py
"""
