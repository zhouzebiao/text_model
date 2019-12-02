# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-11-25 15:
"""

import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, embedding_size, hidden_size, num_heads, attention_dropout):
        if hidden_size % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                    .format(hidden_size, num_heads))

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.embedding_size = embedding_size
        self.q_dense_layer = None
        self.k_dense_layer = None
        self.v_dense_layer = None
        self.output_dense_layer = None

    def build(self, input_shape):
        self.q_dense_layer = tf.keras.layers.Dense(
            self.embedding_size, use_bias=False, name="q")
        self.k_dense_layer = tf.keras.layers.Dense(
            self.embedding_size, use_bias=False, name="k")
        self.v_dense_layer = tf.keras.layers.Dense(
            self.embedding_size, use_bias=False, name="v")
        self.output_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=False, name="output_transform")
        super(Attention, self).build(input_shape)

    # def get_config(self):
    #     return {
    #         "hidden_size": self.hidden_size,
    #         "num_heads": self.num_heads,
    #         "attention_dropout": self.attention_dropout,
    #     }

    def split_heads(self, x):

        length = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        # Calculate depth of last dimension after it has been split.
        depth = (self.embedding_size // self.num_heads)

        # Split the last dimension
        x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

        # Transpose the result
        return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])  # [batch, length, num_heads, depth]
        return tf.reshape(x, [batch_size, length, self.embedding_size])

    def call(self, inputs, **kwargs):
        x = inputs['query_input']  # [batch_size, length_x, hidden_size]
        y = inputs['source_input']  # [batch_size, length_x, hidden_size]
        bias = inputs['bias']  # 缩放点击计算需要，可广播
        training = inputs['training']  # boolean
        cache = inputs['cache']
        # {"k": tensor with shape [batch_size, i, key_channels],
        # "v": tensor with shape [batch_size, i, value_channels]}
        #  i 为当前解码长度.

        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([tf.cast(cache["k"], k.dtype), k], axis=1)
            v = tf.concat([tf.cast(cache["v"], k.dtype), v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 缩放
        q /= (self.hidden_size // self.num_heads) ** -0.5

        # 点积
        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        # float16
        weights = tf.nn.softmax(logits, name="attention_weights")
        if training is not None:
            weights = tf.nn.dropout(weights, rate=self.attention_dropout)
        attention_output = tf.matmul(weights, v)

        attention_output = self.combine_heads(attention_output)

        attention_output = self.output_dense_layer(attention_output)

        def count():
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                # print(shape)
                # print(len(shape))
                variable_parameters = 1
                for dim in shape:
                    # print(dim)
                    variable_parameters *= dim.value
                # print(variable_parameters)
                total_parameters += variable_parameters
            print(total_parameters)

        count()
        return attention_output, cache


class SelfAttention(Attention):

    def call(self, inputs, **kwargs):
        inputs['source_input'] = inputs['query_input']
        return super(SelfAttention, self).call(inputs, **kwargs)
