# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-12-03 09:
"""
import tensorflow as tf


class EmbeddingSharedWeights(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size, hidden_size, initializer_range):
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.shared_weights = None
        self.initializer_range = initializer_range
        # self.data_type = data_type
        self.map_weights = None

    def build(self, input_shape):
        self.shared_weights = tf.compat.v1.get_variable(
            "weights", [self.vocab_size, self.embedding_size],
            initializer=tf.random_normal_initializer(0., self.embedding_size ** -0.5), )

        self.map_weights = tf.compat.v1.get_variable(
            "map_weights", [self.embedding_size, self.hidden_size],
            initializer=tf.random_normal_initializer(0., self.embedding_size ** -0.5), )

    def call(self, inputs, **kwargs):
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

        def count():
            total_parameters = 0
            for variable in tf.compat.v1.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    # print(dim)
                    variable_parameters *= dim.value
                # print(variable_parameters)
                total_parameters += variable_parameters
            print(total_parameters)

        count()
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
