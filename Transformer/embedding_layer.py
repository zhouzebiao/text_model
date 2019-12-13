# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-12-03 09:
"""
import tensorflow as tf


class EmbeddingSharedWeights(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size, data_type):
        super(EmbeddingSharedWeights, self).__init__(dtype=data_type)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.shared_weights = None
        self.data_type = data_type

    def build(self, input_shape):
        self.shared_weights = tf.compat.v1.get_variable(
            "weights", [self.vocab_size, self.embedding_size],
            initializer=tf.random_normal_initializer(0., self.embedding_size ** -0.5), dtype=self.data_type)

    def call(self, inputs, **kwargs):
        mask = tf.cast(tf.not_equal(inputs, 0), dtype=self.data_type)

        embeddings = tf.gather(self.shared_weights, tf.cast(inputs, dtype=tf.int32))
        embeddings *= tf.expand_dims(mask, -1)
        embeddings *= self.embedding_size ** 0.5
        return embeddings

    def linear(self, x):
        """Computes logits by running x through a linear layer.
        Args:
          x: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
          float32 tensor with shape [batch_size, length, vocab_size].
        """

        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        x = tf.reshape(x, [-1, self.embedding_size])
        logits = tf.matmul(x, self.shared_weights, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])
