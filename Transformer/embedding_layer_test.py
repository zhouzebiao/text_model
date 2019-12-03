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
        embedding = 64
        length = 2
        layer = embedding_layer.EmbeddingSharedWeights(vocab_size, embedding)

        inputs = tf.ones([batch_size, length], dtype="int32")
        y = layer(inputs)
        self.assertEqual(y.shape, (batch_size, length, embedding,))


if __name__ == "__main__":
    tf.test.main()
"""
export PYTHONPATH=/data/model;CUDA_VISIBLE_DEVICES='7' python embedding_layer_test.py
"""
