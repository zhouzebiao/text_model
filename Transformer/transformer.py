# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-12-04 17:
"""
import numpy as np

import attention_layer
import beam_search
import embedding_layer
import feed_forward_layer
import metrics
import tensorflow as tf
from official.transformer.utils.tokenizer import EOS_ID


def create_model(params):
    """Creates transformer model."""
    with tf.name_scope("model"):
        if params['train']:
            inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
            targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
            internal_model = Transformer(params)
            logits = internal_model([inputs, targets])
            vocab_size = params["vocab_size"]
            label_smoothing = params["label_smoothing"]
            if params["enable_metrics_in_training"]:
                # logits = metrics.MetricLayer(vocab_size, params['data_type'])([logits, targets])
                logits = metrics.MetricLayer(vocab_size, params['data_type'])([logits, targets])
            logits = tf.keras.layers.Lambda(lambda x: x, name="logits", dtype=params['data_type'])(logits)
            model = tf.keras.Model([inputs, targets], logits)
            loss = metrics.transformer_loss(
                logits, targets, label_smoothing, vocab_size, params['data_type'])
            model.add_loss(loss)
            return model

        else:
            inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
            internal_model = Transformer(params)
            ret = internal_model([inputs])
            outputs, scores = ret["outputs"], ret["scores"]
            return tf.keras.Model(inputs, [outputs, scores])


class Transformer(tf.keras.Model):
    def __init__(self, params):
        super(Transformer, self).__init__(dtype=params['data_type'])
        self.params = params
        self.embedding_layer = embedding_layer.EmbeddingSharedWeights(
            params['vocab_size'], params['embedding_size'], params['data_type'])
        self.encoder_stack = EncoderStack(params)
        self.decoder_stack = DecoderStack(params)
        self._NEG_INF_FP32 = -1e9
        self._NEG_INF_FP16 = np.finfo(np.float16).min
        print(self.params)


    def get_padding_bias(self, x):
        with tf.name_scope("attention_bias"):
            padding = tf.cast(tf.equal(x, 0), self.dtype)
            attention_bias = padding * self._NEG_INF_FP32
            attention_bias = tf.expand_dims(
                tf.expand_dims(attention_bias, axis=1), axis=1)
        return attention_bias

    def get_position_encoding(self, length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
        """Return positional encoding.

        Calculates the position encoding as a mix of sine and cosine functions with
        geometrically increasing wavelengths.
        Defined and formulized in Attention is All You Need, section 3.5.

        Args:
          length: Sequence length.
          hidden_size: Size of the
          min_timescale: Minimum scale that will be applied at each position
          max_timescale: Maximum scale that will be applied at each position

        Returns:
          Tensor with shape [length, hidden_size]
        """
        # We compute the positional encoding in float32 even if the model uses
        # float16, as many of the ops used, like log and exp, are numerically unstable
        # in float16.
        position = tf.cast(tf.range(length), self.dtype)
        num_timescales = hidden_size // 2
        log_timescale_increment = (
                tf.math.log(tf.cast(max_timescale, self.dtype) / tf.cast(min_timescale, self.dtype)) / (
                tf.cast(num_timescales, self.dtype) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), self.dtype) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        return signal

    def get_decoder_self_attention_bias(self, length):
        """Calculate bias for decoder that maintains model's autoregressive property.


        Returns:
          float tensor of shape [1, 1, length, length]
        """
        neg_inf = self._NEG_INF_FP16 if self.dtype == tf.float16 else self._NEG_INF_FP32
        with tf.name_scope("decoder_self_attention_bias"):
            valid_loc = tf.linalg.band_part(tf.ones([length, length], dtype=self.dtype), -1, 0)
            valid_loc = tf.reshape(valid_loc, [1, 1, length, length])
            decoder_bias = neg_inf * (1.0 - valid_loc)
        return decoder_bias

    def call(self, inputs, **kwargs):
        """Calculate target logits or inferred target sequences.
        Args:
          inputs: input tensor list of size 1 or 2.
            First item, inputs: int tensor with shape [batch_size, input_length].
            Second item (optional), targets: None or int tensor with shape
              [batch_size, target_length]
        Returns:
          If targets is defined, then return logits for each word in the target
          sequence. float tensor with shape [batch_size, target_length, vocab_size]
          If target is none, then generate output sequence one token at a time.
            returns a dictionary {
              outputs: [batch_size, decoded length]
              scores: [batch_size, float]}
          Even when float16 is used, the output tensor(s) are always float32.
        """
        if len(inputs) == 2:
            inputs, targets = inputs[0], inputs[1]
        else:
            inputs, targets = inputs[0], None
        training = self.params['train']
        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("Transformer"):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = self.get_padding_bias(inputs)

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(inputs, attention_bias, training)
            # Generate output sequence if targets is None, or return logits if target
            # sequence is known.
            if targets is None:
                print('if targets is None:',encoder_outputs,attention_bias)
                logits = self.predict(
                    {'encoder_outputs': encoder_outputs, 'encoder_decoder_attention_bias': attention_bias})
                return logits
            else:
                logits = self.decode(targets, encoder_outputs, attention_bias, training)
                return logits

    def encode(self, inputs, attention_bias, training):
        """Generate continuous representation for inputs.
        Args:
          inputs: int tensor with shape [batch_size, input_length].
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
          training: boolean, whether in training mode or not.
        Returns:
          float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            embedded_inputs = self.embedding_layer(inputs)
            embedded_inputs = tf.cast(embedded_inputs, self.dtype)
            attention_bias = tf.cast(attention_bias, self.dtype)

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = self.get_position_encoding(
                    length, self.params["embedding_size"])
                pos_encoding = tf.cast(pos_encoding, self.dtype)

                encoder_inputs = embedded_inputs + pos_encoding

            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=self.params["layer_postprocess_dropout"])
            return self.encoder_stack({'encoder_inputs': encoder_inputs, 'attention_bias': attention_bias})

    def decode(self, targets, encoder_outputs, attention_bias, training):
        """Generate logits for each value in the target sequence.
        Args:
          targets: target values for the output sequence. int tensor with shape
            [batch_size, target_length]
          encoder_outputs: continuous representation of input sequence. float tensor
            with shape [batch_size, input_length, hidden_size]
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
          training: boolean, whether in training mode or not.
        Returns:
          float32 tensor with shape [batch_size, target_length, vocab_size]
        """

        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            decoder_inputs = self.embedding_layer(targets)

            decoder_inputs = tf.cast(decoder_inputs, self.dtype)
            attention_bias = tf.cast(attention_bias, self.dtype)
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                pos_encoding = self.get_position_encoding(length, self.params["hidden_size"])
                pos_encoding = tf.cast(pos_encoding, self.dtype)
                decoder_inputs += pos_encoding
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=self.params["layer_postprocess_dropout"])

            # Run values
            decoder_self_attention_bias = self.get_decoder_self_attention_bias(length)
            outputs = self.decoder_stack(
                {'decoder_inputs': decoder_inputs,
                 'encoder_outputs': encoder_outputs,
                 'decoder_self_attention_bias': decoder_self_attention_bias,
                 'attention_bias': attention_bias,
                 'cache': None})
            logits = self.embedding_layer.linear(outputs)

            logits = tf.cast(logits, self.dtype)
            return logits

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = self.get_position_encoding(
            max_decode_length + 1, self.params["hidden_size"])
        timing_signal = tf.cast(timing_signal, self.dtype)
        decoder_self_attention_bias = self.get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.
            Args:
              ids: Current decoded sequences. int tensor with shape [batch_size *
                beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.
            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_inputs = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_inputs = self.embedding_layer(decoder_inputs)
            decoder_inputs += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack({
                'decoder_inputs': decoder_inputs,
                'encoder_outputs': cache.get("encoder_outputs"),
                'decoder_self_attention_bias': self_attention_bias,
                'attention_bias': cache.get("encoder_decoder_attention_bias"),
                'cache': cache})
            logits = self.embedding_layer.linear(decoder_outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def predict(self, inputs, **kwargs):
        """Return predicted sequence."""
        encoder_outputs = inputs['encoder_outputs']
        encoder_decoder_attention_bias = inputs['encoder_decoder_attention_bias']
        batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params["extra_decode_length"]
        encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias, self.dtype)

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        # Create cache storing decoder attention values for each layer.
        # pylint: disable=g-complex-comprehension
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params["embedding_size"]], dtype=self.dtype),
                "v": tf.zeros([batch_size, 0, self.params["embedding_size"]], dtype=self.dtype)
            } for layer in range(self.params["num_hidden_layers"])
        }
        # pylint: enable=g-complex-comprehension

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.params["vocab_size"],
            beam_size=self.params["beam_size"],
            alpha=self.params["alpha"],
            max_decode_length=max_decode_length,
            eos_id=EOS_ID,
            data_type=self.dtype)

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(EncoderStack, self).__init__(dtype=params['data_type'])
        self.params = params
        self.scale = None
        self.bias = None
        self.postprocess_dropout = None
        self.layers = []

    def build(self, input_shape):
        """Builds the encoder stack."""
        params = self.params
        self.scale = self.add_weight("layer_norm_scale", shape=[params['embedding_size']],
                                     initializer=tf.ones_initializer())
        self.bias = self.add_weight("layer_norm_bias", shape=[params['embedding_size']],
                                    initializer=tf.zeros_initializer())
        self.postprocess_dropout = params["layer_postprocess_dropout"]
        for _ in range(params["num_hidden_layers"]):
            self_attention_layer = attention_layer.SelfAttention(params['hidden_size'], params['num_heads'],
                                                                 params['attention_dropout'], params['embedding_size'],
                                                                 params['train'], self.dtype)
            feed_forward_network = feed_forward_layer.FeedForwardNetwork(
                params["hidden_size"], params["filter_size"], params["relu_dropout"], params['train'],
                self.dtype)

            self.layers.append([
                self_attention_layer,
                feed_forward_network
            ])

        super(EncoderStack, self).build(input_shape)

    def layer_norm(self, decoder_inputs, epsilon=1e-6):
        mean = tf.reduce_mean(decoder_inputs, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(decoder_inputs - mean), axis=[-1], keepdims=True)
        norm_x = (decoder_inputs - mean) * tf.math.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias

    def shortcut_connection(self, x, y):
        if self.params['train']:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y

    def call(self, inputs, **kwargs):
        encoder_inputs = inputs['encoder_inputs']  # [batch_size, input_length, hidden_size]
        attention_bias = inputs['attention_bias']  # [batch_size, 1,1, input_length]
        for n, (self_attention, feed_forward) in enumerate(self.layers):
            query_input = self.layer_norm(encoder_inputs)
            output, _ = self_attention(
                {'query_input': query_input, 'bias': attention_bias, 'cache': None})
            encoder_inputs = self.shortcut_connection(encoder_inputs, output)
            feed_inputs = self.layer_norm(encoder_inputs)
            output = feed_forward(feed_inputs)
            encoder_inputs = self.shortcut_connection(encoder_inputs, output)
        return self.layer_norm(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):

    def __init__(self, params):
        super(DecoderStack, self).__init__(dtype=params['data_type'])
        self.params = params
        self.scale = None
        self.bias = None
        self.postprocess_dropout = None
        self.layers = []

    def build(self, input_shape):
        """Builds the decoder stack."""
        params = self.params
        self.scale = self.add_weight("layer_norm_scale", shape=[params['hidden_size']],
                                     initializer=tf.ones_initializer())
        self.bias = self.add_weight("layer_norm_bias", shape=[params['hidden_size']],
                                    initializer=tf.zeros_initializer())
        self.postprocess_dropout = params["layer_postprocess_dropout"]
        for _ in range(params["num_hidden_layers"]):
            self_attention_layer = attention_layer.SelfAttention(params['hidden_size'], params['num_heads'],
                                                                 params['attention_dropout'], params['embedding_size'],
                                                                 params['train'], self.dtype)
            enc_dec_attention_layer = attention_layer.Attention(params['hidden_size'], params['num_heads'],
                                                                params['attention_dropout'], params['embedding_size'],
                                                                params['train'], self.dtype)
            feed_forward_network = feed_forward_layer.FeedForwardNetwork(
                params["hidden_size"], params["filter_size"], params["relu_dropout"], params['train'], self.dtype)

            self.layers.append([
                self_attention_layer,
                enc_dec_attention_layer,
                feed_forward_network,
            ])
        super(DecoderStack, self).build(input_shape)

    def layer_norm(self, decoder_inputs, epsilon=1e-6):
        mean = tf.reduce_mean(decoder_inputs, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(decoder_inputs - mean), axis=[-1], keepdims=True)
        norm_x = (decoder_inputs - mean) * tf.math.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias

    def shortcut_connection(self, x, y):
        if self.params['train']:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y

    def call(self, inputs, **kwargs):
        decoder_inputs = inputs['decoder_inputs']  # [batch_size, target_length, hidden_size]
        encoder_outputs = inputs['encoder_outputs']  # [batch_size, input_length, hidden_size]
        decoder_self_attention_bias = inputs['decoder_self_attention_bias']  # [1, 1, target_len, target_length]
        attention_bias = inputs['attention_bias']  # [batch_size, 1, 1, input_length]
        cache = inputs['cache']
        for n, (self_attention, enc_dec_attention, feed_forward) in enumerate(self.layers):
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            query_input = self.layer_norm(decoder_inputs)
            output, layer_cache = self_attention(
                {'query_input': query_input, 'bias': decoder_self_attention_bias, 'cache': layer_cache})
            decoder_inputs = self.shortcut_connection(decoder_inputs, output)
            query_input = self.layer_norm(decoder_inputs)
            output, _ = enc_dec_attention(
                {'query_input': query_input, 'source_input': encoder_outputs, 'bias': attention_bias,
                 'cache': None})
            decoder_inputs = self.shortcut_connection(decoder_inputs, output)
            query_input = self.layer_norm(decoder_inputs)
            output = feed_forward(query_input)
            decoder_inputs = self.shortcut_connection(decoder_inputs, output)
        return self.layer_norm(decoder_inputs)
