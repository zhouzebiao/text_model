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
        # self.vocab_size = vocab_size
        self.num_labels = 2
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
            self.shared_weights = self.add_weight(
                "weights",
                shape=[self.num_labels, self.hidden_size],
                initializer=tf.random_normal_initializer(
                    mean=0., stddev=self.hidden_size ** -0.5))
        super(EmbeddingSharedWeights, self).build(input_shape)


    def call(self, inputs, mode="embedding"):
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs):
        """Applies embedding based on inputs tensor."""
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, length]
            embeddings = tf.gather(self.shared_weights, inputs)
            mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
            embeddings *= tf.expand_dims(mask, -1)
            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5

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
            # length = tf.shape(inputs)[1]

            x = tf.reshape(inputs, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, self.num_labels])
