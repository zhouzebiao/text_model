# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-12-03 09:
"""

import tensorflow as tf


class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, embedding_size, hidden_size, initializer_range):
        """Specify characteristic parameters of embedding layer.

        Args:
          vocab_size: Number of tokens in the embedding. (Typically ~32,000)
          hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
        """
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.shared_weights = None
        self.initializer_range = initializer_range
        # self.data_type = data_type
        self.map_weights = None

    def build(self, input_shape):
        """Build embedding layer."""
        with tf.name_scope("embedding_and_softmax"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.shared_weights = tf.compat.v1.get_variable(
                "weights", [self.vocab_size, self.embedding_size],
                initializer=tf.random_normal_initializer(
                    mean=0., stddev=self.hidden_size ** -0.5))

            self.map_weights = tf.compat.v1.get_variable(
                "map_weights", [self.embedding_size, self.hidden_size],
                initializer=tf.random_normal_initializer(
                    mean=0., stddev=self.hidden_size ** -0.5))

        super(EmbeddingSharedWeights, self).build(input_shape)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
        }

    def call(self, inputs, mode="embedding"):
        """Get token embeddings of inputs.

        Args:
          inputs: An int64 tensor with shape [batch_size, length]
          mode: string, a valid value is one of "embedding" and "linear".
        Returns:
          outputs: (1) If mode == "embedding", output embedding tensor, float32 with
            shape [batch_size, length, embedding_size]; (2) mode == "linear", output
            linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
          ValueError: if mode is not valid.
        """
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs):
        """Applies embedding based on inputs tensor."""
        with tf.name_scope("embedding"):
            print('EmbeddingSharedWeights', self.shared_weights)
            embeddings = tf.gather(self.shared_weights, tf.cast(inputs, dtype=tf.int32))
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]
            mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
            embeddings *= tf.expand_dims(mask, -1)
            embeddings *= self.embedding_size ** 0.5
            # 2. project vector(output_middle) to the hidden space
            embeddings = tf.matmul(embeddings, self.map_weights)
            # embeddings=[batch_size, sequence_length, hidden_size]
            embeddings = tf.reshape(embeddings, (batch_size, length, self.hidden_size))

            return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.

        Args:
          inputs: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
          float32 tensor with shape [batch_size, length, vocab_size].
        """
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]

            x = tf.reshape(inputs, [-1, self.hidden_size])
            x = tf.matmul(x, self.map_weights,transpose_b=True)
            print('_linear',x)
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])
