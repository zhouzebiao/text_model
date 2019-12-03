# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-12-03 09:
"""
import tensorflow as tf


class EmbeddingSharedWeights(tf.layers.Layer):

    def __init__(self, vocab_size, embedding_size):
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.shared_weights = None

    def build(self, input_shape):
        self.shared_weights = tf.get_variable(
            "weights", [self.vocab_size, self.embedding_size],
            initializer=tf.random_normal_initializer(
                0., self.embedding_size ** -0.5))

    def call(self, inputs, **kwargs):
        mask = tf.to_float(tf.not_equal(inputs, 0))

        embeddings = tf.gather(self.shared_weights, tf.cast(inputs, dtype=tf.int32))
        embeddings *= tf.expand_dims(mask, -1)
        embeddings *= self.embedding_size ** 0.5
        return embeddings
