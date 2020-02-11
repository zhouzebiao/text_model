# -*- coding: utf-8 -*-
"""
 Created by zaber on 2020--01-03 09:
"""
import attention_layer
import embedding_layer
import feed_forward_layer
import tensorflow as tf
from official.transformer.model import model_utils


class Config(object):
    def __init__(self):
        self.meta_data = {}
        self.generate_threshold = 327
        self.generate_DATA_MIN_COUNT = 6
        self.generate_search = False
        self.vocab_size = 32768
        self.max_seq_length = 256
        self.num_labels = 2
        self.activation = 'relu'
        self.last_activation = 'sigmoid'
        self.hidden_size = 512
        self.initializer_range = 0.2
        self.dropout_keep_prob = 0.35


def get_loss_fn(batch_size, labels, logits):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)

    def loss_fn(labels, logits):
        print(labels.shape, logits.shape, 'def loss_fn(labels, logits):')
        with tf.name_scope("Transformer"):
            per_example_loss = loss_object(labels, logits)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

    loss = loss_fn(labels, logits)
    return loss


def create_model(params):
    """Creates transformer model."""
    with tf.name_scope("model"):
        core_model = Transformer(params)
        return core_model


class Transformer(tf.keras.Model):
    """Transformer model with Keras.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continuous
    representation, and the decoder uses the encoder output to generate
    probabilities for the output sequence.
    """

    def __init__(self, params, name=None):
        """Initialize layers to build Transformer model.

        Args:
          params: hyperparameter object defining layer sizes, dropout values, etc.
          name: name of the model.
        """
        super(Transformer, self).__init__(name=name)
        self.params = params
        self.embedding_layer = embedding_layer.EmbeddingSharedWeights(
            params["vocab_size"], params["embedding_size"], params["hidden_size"], 0.02)
        self.encoder_stack1 = EncoderStack(params)
        self.encoder_stack2 = EncoderStack(params)
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.output_dense = tf.keras.layers.Dense(units=2, activation=tf.nn.tanh)
        # self.decoder_stack = DecoderStack(params)

    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self, inputs, **kwargs):
        inputs, targets = inputs[0], inputs[1]
        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("Transformer"):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias1 = model_utils.get_padding_bias(inputs)
            attention_bias2 = model_utils.get_padding_bias(targets)

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            logits1 = self.encode(inputs, attention_bias1, self.params['train'])
            # Generate output sequence if targets is None, or return logits if target
            # sequence is known.
            logits2 = self.decode(targets, attention_bias2, self.params['train'])
            # logits = tf.matmul(logits1, logits2, transpose_b=True)
            # logits = tf.einsum("BTNH,BFNH->BNFT", logits1, logits2)
            print(logits1.shape, logits2.shape, 'logits1.shape,logits2.shape')
            x = self.concatenate([logits1, logits2])
            logits = self.output_dense(x)
            print(logits.shape, 'logits.shape')
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
            embedded_inputs = tf.cast(embedded_inputs, self.params["dtype"])
            inputs_padding = model_utils.get_padding(inputs)
            attention_bias = tf.cast(attention_bias, self.params["dtype"])

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params["hidden_size"])
                pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
                encoder_inputs = embedded_inputs + pos_encoding

            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=self.params["layer_postprocess_dropout"])

            outputs = self.encoder_stack1(
                encoder_inputs, attention_bias, inputs_padding)
            # logits = self.embedding_layer(outputs, mode="linear")
            logits = tf.reduce_sum(outputs, axis=1)
            logits = tf.reshape(logits, [self.params["batch_size"], self.params["hidden_size"]])
            logits = tf.cast(logits, tf.float32)
            return logits

    def decode(self, targets, attention_bias, training):
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
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            embedded_inputs = self.embedding_layer(targets)
            embedded_inputs = tf.cast(embedded_inputs, self.params["dtype"])
            inputs_padding = model_utils.get_padding(targets)
            attention_bias = tf.cast(attention_bias, self.params["dtype"])

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params["hidden_size"])
                pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
                encoder_inputs = embedded_inputs + pos_encoding

            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=self.params["layer_postprocess_dropout"])

            outputs = self.encoder_stack1(
                encoder_inputs, attention_bias, inputs_padding)
            # logits = self.embedding_layer(outputs, mode="linear")
            logits = tf.reduce_sum(outputs, axis=1)
            logits = tf.reshape(logits, [self.params["batch_size"], self.params["hidden_size"]])
            logits = tf.cast(logits, tf.float32)
            return logits

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = model_utils.get_position_encoding(
            max_decode_length + 1, self.params["hidden_size"])
        timing_signal = tf.cast(timing_signal, self.params["dtype"])
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
            max_decode_length, dtype=self.params["dtype"])

        # TODO(b/139770046): Refactor code with better naming of i.
        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.

            Args:
              ids: Current decoded sequences. int tensor with shape [batch_size *
                beam_size, i + 1].
              i: Loop index.
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.

            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_layer(decoder_input)

            if self.params["padded_decode"]:
                timing_signal_shape = timing_signal.shape.as_list()
                decoder_input += tf.slice(timing_signal, [i, 0],
                                          [1, timing_signal_shape[1]])

                bias_shape = decoder_self_attention_bias.shape.as_list()
                self_attention_bias = tf.slice(
                    decoder_self_attention_bias, [0, 0, i, 0],
                    [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
            else:
                decoder_input += timing_signal[i:i + 1]

                self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            decoder_outputs = self.decoder_stack(
                decoder_input,
                cache.get("encoder_outputs"),
                self_attention_bias,
                cache.get("encoder_decoder_attention_bias"),
                training=training,
                cache=cache)
            logits = self.embedding_layer(decoder_outputs, mode="linear")
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn


class LayerNormalization(tf.keras.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        # Pass dtype=float32, as we have not yet tested if layer norm is numerically
        # stable in float16 and bfloat16.
        super(LayerNormalization, self).__init__(dtype="float32")
        self.hidden_size = hidden_size

    def build(self, input_shape):
        """Builds the layer."""
        self.scale = self.add_weight(
            "layer_norm_scale",
            shape=[self.hidden_size],
            initializer=tf.ones_initializer())
        self.bias = self.add_weight(
            "layer_norm_bias",
            shape=[self.hidden_size],
            initializer=tf.zeros_initializer())
        super(LayerNormalization, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
        }

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params):
        super(PrePostProcessingWrapper, self).__init__()
        self.layer = layer
        self.params = params
        self.postprocess_dropout = params["layer_postprocess_dropout"]

    def build(self, input_shape):
        # Create normalization layer
        self.layer_norm = LayerNormalization(self.params["hidden_size"])
        super(PrePostProcessingWrapper, self).build(input_shape)

    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = self.params['train']

        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if training:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y


class EncoderStack(tf.keras.layers.Layer):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.params = params
        self.self_attention_layer = None
        self.feed_forward_network = None
        self.output_normalization = None

    def build(self, input_shape):
        """Builds the encoder stack."""
        params = self.params
        self_attention_layer = attention_layer.SelfAttention(
            params["hidden_size"], params["num_heads"],
            params["attention_dropout"])
        feed_forward_network = feed_forward_layer.FeedForwardNetwork(
            params["hidden_size"], params["filter_size"], params["relu_dropout"], params['train'])
        self.self_attention_layer = PrePostProcessingWrapper(self_attention_layer, params)
        self.feed_forward_network = PrePostProcessingWrapper(feed_forward_network, params)

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params["hidden_size"])
        super(EncoderStack, self).build(input_shape)

    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
            1, input_length]
          inputs_padding: tensor with shape [batch_size, input_length], inputs with
            zero paddings.

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for _ in range(self.params["num_hidden_layers"]):
            # Run inputs through the sublayers.

            with tf.compat.v1.variable_scope("encode_layer", reuse=tf.compat.v1.AUTO_REUSE):
                with tf.compat.v1.variable_scope("self_attention", reuse=tf.compat.v1.AUTO_REUSE):
                    encoder_inputs = self.self_attention_layer(
                        encoder_inputs, attention_bias, training=self.params['train'])
                with tf.compat.v1.variable_scope("ffn", reuse=tf.compat.v1.AUTO_REUSE):
                    encoder_inputs = self.feed_forward_network(encoder_inputs)

        return self.output_normalization(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):
    """Transformer decoder stack.

    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer is composed of the sublayers:
      1. Self-attention layer
      2. Multi-headed attention layer combining encoder outputs with results from
         the previous self-attention layer.
      3. Feedforward network (2 fully-connected layers)
    """

    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.params = params
        self.self_attention_layer = None
        self.enc_dec_attention_layer = None
        self.feed_forward_network = None
        self.output_normalization = None

    def build(self, input_shape):
        """Builds the decoder stack."""
        params = self.params
        self_attention_layer = attention_layer.SelfAttention(
            params["hidden_size"], params["num_heads"],
            params["attention_dropout"])
        enc_dec_attention_layer = attention_layer.Attention(
            params["hidden_size"], params["num_heads"],
            params["attention_dropout"])
        feed_forward_network = feed_forward_layer.FeedForwardNetwork(
            params["hidden_size"], params["filter_size"], params["relu_dropout"], 0.02)
        self.self_attention_layer = PrePostProcessingWrapper(self_attention_layer, params)
        self.enc_dec_attention_layer = PrePostProcessingWrapper(enc_dec_attention_layer, params)
        self.feed_forward_network = PrePostProcessingWrapper(feed_forward_network, params)
        self.output_normalization = LayerNormalization(params["hidden_size"])
        super(DecoderStack, self).build(input_shape)

    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self,
             decoder_inputs,
             encoder_outputs,
             decoder_self_attention_bias,
             attention_bias,
             training,
             cache=None):
        """Return the output of the decoder layer stacks.

        Args:
          decoder_inputs: A tensor with shape
            [batch_size, target_length, hidden_size].
          encoder_outputs: A tensor with shape
            [batch_size, input_length, hidden_size]
          decoder_self_attention_bias: A tensor with shape
            [1, 1, target_len, target_length], the bias for decoder self-attention
            layer.
          attention_bias: A tensor with shape [batch_size, 1, 1, input_length],
            the bias for encoder-decoder attention layer.
          training: A bool, whether in training mode or not.
          cache: (Used for fast decoding) A nested dictionary storing previous
            decoder self-attention values. The items are:
              {layer_n: {"k": A tensor with shape [batch_size, i, key_channels],
                         "v": A tensor with shape [batch_size, i, value_channels]},
                           ...}
          decode_loop_step: An integer, the step number of the decoding loop. Used
            only for autoregressive inference on TPU.

        Returns:
          Output of decoder layer stack.
          float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        for n in range(self.params["num_hidden_layers"]):
            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.compat.v1.variable_scope("decode_layer", reuse=tf.compat.v1.AUTO_REUSE):
                with tf.compat.v1.variable_scope("self_attention", reuse=tf.compat.v1.AUTO_REUSE):
                    decoder_inputs = self.self_attention_layer(
                        decoder_inputs,
                        decoder_self_attention_bias,
                        training=training,
                        cache=layer_cache)
                with tf.compat.v1.variable_scope("encdec_attention", reuse=tf.compat.v1.AUTO_REUSE):
                    decoder_inputs = self.enc_dec_attention_layer(
                        decoder_inputs,
                        encoder_outputs,
                        attention_bias,
                        training=training)
                with tf.name_scope("ffn"):
                    decoder_inputs = self.feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)