# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-09-14 12:
"""
from abc import ABC

import tensorflow as tf
import copy
import six
import json


def get_model(config, float_type):
    core_model = TextRNN(config, float_type)
    return core_model


class Config(object):
    def __init__(self):
        self.meta_data = {}
        self.generate_threshold = 327
        self.generate_DATA_MIN_COUNT = 6
        self.generate_search = False
        self.vocab_size = 32768
        self.max_seq_length = 256
        self.num_labels = 2
        self.activation = 'tanh'
        self.last_activation = 'sigmoid'
        self.rnn_units = 'LSTM'
        self.hidden_size = 256
        self.num_units = 64
        self.initializer_range = 0.2
        self.dropout_keep_prob = 0.35

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = Config()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TextRNN(tf.keras.layers.Layer, ABC):
    def __init__(self, config, float_type=tf.float32, **kwargs):
        super(TextRNN, self).__init__(**kwargs)
        self.config = copy.deepcopy(config)
        self.float_type = float_type
        self.embedding_lookup = None
        self.bi_rnn = None

    def build(self, input_shape):
        self.embedding_lookup = EmbeddingLookup(
            vocab_size=self.config.vocab_size,
            embedding_size=self.config.hidden_size,
            initializer_range=self.config.initializer_range,
            dtype=tf.float32,
            name="word_embeddings")
        self.bi_rnn = BiRNN(rnn_units=self.config.rnn_units,
                            num_units=self.config.num_units,
                            activation=self.config.activation,
                            num_labels=self.config.num_labels,
                            last_activation=self.config.last_activation,
                            dropout_keep_prob=self.config.dropout_keep_prob
                            )

    def __call__(self, inputs, **kwargs):
        return super(TextRNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        x = self.embedding_lookup(inputs)
        output = self.bi_rnn(x)
        return output


class BiRNN(tf.keras.layers.Layer, ABC):
    def __init__(self, rnn_units, num_units, activation, num_labels, last_activation, dropout_keep_prob, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.rnn_units = rnn_units
        self.num_units = num_units
        self.activation = activation
        self.num_labels = num_labels
        self.last_activation = last_activation
        self.dropout_keep_prob = dropout_keep_prob
        self.layer = None
        self.dropout = None
        self.output_dense = None

    def build(self, input_shape):
        if self.rnn_units == 'LSTM':
            layer_cell = tf.keras.layers.LSTM(self.num_units, activation=self.activation)
        elif self.rnn_units == 'GRU':
            layer_cell = tf.keras.layers.GRU(self.num_units, activation=self.activation)
        else:
            layer_cell = tf.keras.layers.RNN(self.num_units, activation=self.activation)
        self.layer = tf.keras.layers.Bidirectional(layer_cell, merge_mode='concat')
        self.dropout = tf.keras.layers.Dropout(self.dropout_keep_prob)
        self.output_dense = tf.keras.layers.Dense(units=self.num_labels, activation=self.last_activation)
        super(BiRNN, self).build(input_shape)

    def __call__(self, inputs, **kwargs):
        return super(BiRNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        x = self.layer(inputs)
        print('#####get_shape_list(x)')
        # x = self.dropout(x)
        x = x[:, -1:]
        print(get_shape_list(x))
        output = self.output_dense(x)
        return output


class EmbeddingLookup(tf.keras.layers.Layer, ABC):
    def __init__(self, vocab_size, embedding_size, initializer_range, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range
        self.embeddings = None

    def build(self, input_shape):
        self.embeddings = self.add_weight("embeddings", shape=[self.vocab_size, self.embedding_size],
                                          initializer=tf.keras.initializers.TruncatedNormal(
                                              stddev=self.initializer_range), dtype=tf.float32)
        super(EmbeddingLookup, self).build(input_shape)

    def __call__(self, inputs, **kwargs):
        return super(EmbeddingLookup, self).__call__(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        input_shape = get_shape_list(inputs)
        flat_input = tf.reshape(inputs, [-1])
        output = tf.gather(self.embeddings, flat_input)
        output = tf.reshape(output, input_shape + [self.embedding_size])
        return output


def get_shape_list(tensor):
    t_shape = tensor.shape.as_list()
    none_static_indexes = []
    for (i, dim) in enumerate(t_shape):
        if dim is None:
            none_static_indexes.append(i)

    if not none_static_indexes:
        return t_shape

    dyn_shape = tf.shape(tensor)
    for i in none_static_indexes:
        t_shape[i] = dyn_shape[i]
    return t_shape
