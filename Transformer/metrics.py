# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-12-04 17:
"""
import functools

import tensorflow as tf


def _pad(a, b):
    x_length = tf.shape(a)[1]
    y_length = tf.shape(b)[1]

    max_length = tf.maximum(x_length, y_length)
    a = tf.pad(a, [[0, 0], [0, max_length - x_length], [0, 0]])
    b = tf.pad(b, [[0, 0], [0, max_length - y_length]])
    return a, b


def cross_entropy_loss(logits, targets, smoothing, vocab_size, data_type):
    logits, targets = _pad(logits, targets)
    confidence = tf.cast(1.0 - smoothing, data_type)
    low_confidence = (1.0 / confidence) / tf.cast(vocab_size - 1, data_type)
    soft_targets = tf.one_hot(tf.cast(targets, tf.int32), depth=vocab_size,
                              on_value=confidence, off_value=low_confidence)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=soft_targets, logits=logits)
    loss = tf.cast(loss, data_type)
    normalizing_constant = -(
            confidence * tf.math.log(confidence) +
            tf.cast(vocab_size - 1, data_type) * low_confidence * tf.math.log(low_confidence + 1e-20))
    loss -= normalizing_constant
    weights = tf.cast(tf.not_equal(targets, 0), data_type)
    return loss * weights, weights


def accuracy(logits, targets, data_type):
    logits, targets = _pad(logits, targets)
    weights = tf.cast(tf.not_equal(targets, 0), data_type)
    outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    padded_labels = tf.cast(targets, tf.int32)
    return tf.cast(tf.equal(outputs, padded_labels), data_type), weights


def sequence_accuracy(logits, targets, data_type):
    logits, targets = _pad(logits, targets)
    targets = tf.cast(targets, tf.int32)
    weights = tf.cast(tf.not_equal(targets, 0), data_type)
    outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    not_correct = tf.cast(tf.not_equal(outputs, targets), data_type) * weights
    axis = list(range(1, len(outputs.get_shape())))
    correct = 1.0 - tf.minimum(tf.cast(1.0, dtype=tf.float16), tf.reduce_sum(not_correct, axis=axis))
    return correct, tf.constant(1.0)


def neg_log_perplexity(logits, targets, vocab_size, data_type):
    num, den = cross_entropy_loss(logits, targets, 0, vocab_size, data_type)
    return -num, den


class MetricLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, data_type):
        super(MetricLayer, self).__init__()
        self.vocab_size = vocab_size
        self.data_type = data_type
        self.metric = None

    def build(self, input_shape):
        acc = functools.partial(accuracy, data_type=self.data_type)
        acc_per_sequence = functools.partial(sequence_accuracy, data_type=self.data_type)
        perplexity = functools.partial(neg_log_perplexity, vocab_size=self.vocab_size, data_type=self.data_type)
        self.metric = [(tf.keras.metrics.Mean('accuracy'), acc),
                       (tf.keras.metrics.Mean('accuracy_per_sequence'), acc_per_sequence),
                       (tf.keras.metrics.Mean('neg_log_perplexity'), perplexity)]
        super(MetricLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        logits, targets = inputs[0], inputs[1]
        for mean, fn in self.metric:
            m = mean(*fn(logits, targets))
            self.add_metric(m)
        return logits


def transformer_loss(logits, labels, smoothing, vocab_size, data_type):
    """Calculates total loss containing cross entropy with padding ignored.
      logits: Tensor of size [batch_size, length_logits, vocab_size]
      labels: Tensor of size [batch_size, length_labels]
      smoothing: Label smoothing constant, used to determine the on and off values
      vocab_size: int size of the vocabulary
    """
    print('transformer_loss',logits, labels)
    entropy, weights = cross_entropy_loss(logits, labels, smoothing, vocab_size, data_type)
    return tf.reduce_sum(entropy) / tf.reduce_sum(weights)
