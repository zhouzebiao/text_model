# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-12-05 15:
"""
import attention_layer
import embedding_layer
import feed_forward_layer
import tensorflow as tf


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, config):
        super(EncoderStack, self).__init__()
        self.config = config
        self.scale = None
        self.bias = None
        self.postprocess_dropout = None
        self.layers = []

    def build(self, input_shape):
        self.scale = self.add_weight("layer_norm_scale", shape=[self.hidden_size], initializer=tf.ones_initializer())
        self.bias = self.add_weight("layer_norm_bias", shape=[self.hidden_size], initializer=tf.zeros_initializer())
        self.postprocess_dropout = self.config["layer_postprocess_dropout"]
        for _ in range(self.config['num_hidden_layers']):
            self_attention = attention_layer.SelfAttention(self.config['embedding_size'],
                                                           self.config['hidden_size'], self.config['num_heads'],
                                                           self.config['attention_dropout'], self.config['train'])
            feed_forward = feed_forward_layer.FeedForwardNetwork(self.config['hidden_size'], self.config['filter_size'],
                                                                 self.config['relu_dropout'], self.config['train'])
            self.layers.append([self_attention, feed_forward])

    def layer_norm(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias

    def shortcut_connection(self, x, y):
        if self.config['train']:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y

    def call(self, inputs, **kwargs):
        encoder_inputs = inputs['encoder_inputs']  # [batch_size, input_length, hidden_size]
        attention_bias = inputs['attention_bias']  # [batch_size, 1,1, input_length]
        with tf.name_scope("encoder"):
            for lid, (self_attention, feed_forward) in enumerate(self.layers):
                query_input = self.layer_norm(encoder_inputs)
                output, _ = self_attention({'query_input': query_input, 'bias': attention_bias})
                encoder_inputs = self.shortcut_connection(encoder_inputs, output)
                feed_inputs = self.layer_norm(encoder_inputs)
                output = feed_forward(feed_inputs)
                encoder_inputs = self.shortcut_connection(encoder_inputs, output)
        return self.layer_norm(encoder_inputs)


class Transformer(tf.keras.Model):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.embedding_layer = embedding_layer.EmbeddingSharedWeights(config['vocab_size'], config['embedding_size'])
        # self.encoder_stack=EncoderStack(conf)
