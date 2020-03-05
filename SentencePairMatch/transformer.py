# -*- coding: utf-8 -*-
"""
 Created by zaber on 2020--01-03 09:
"""
import attention_layer
import embedding_layer
import feed_forward_layer
import tensorflow as tf
from official.transformer.model import model_utils
from official.transformer.v2 import metrics


def get_loss_fn(num_classes, loss_factor=1.0):
    """Gets the classification loss function."""

    def classification_loss_fn(labels, logits):
        """Classification loss."""
        labels = tf.squeeze(labels)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(
            tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(
            tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        loss *= loss_factor
        return loss

    return classification_loss_fn


def create_model(params, is_train):
    """Creates transformer model."""
    with tf.name_scope("model"):
        if is_train:
            inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
            targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
            labels = tf.keras.layers.Input((None,), dtype="int64", name="labels")
            internal_model = Transformer(params, name="transformer_v2")
            logits = internal_model([inputs, targets, labels], training=is_train)
            # vocab_size = params["vocab_size"]
            num_labels = 2
            label_smoothing = params["label_smoothing"]
            if params["enable_metrics_in_training"]:
                logits = metrics.MetricLayer(num_labels)([logits, labels])
            logits = tf.keras.layers.Lambda(lambda x: x, name="logits",
                                            dtype=tf.float32)(logits)
            model = tf.keras.Model([inputs, targets, labels], logits)
            # TODO(reedwm): Can we do this loss in float16 instead of float32?
            # loss = metrics.transformer_loss(
            #     logits, labels, label_smoothing, num_labels)
            loss = get_loss_fn(2, loss_factor=1.0)(labels, logits)
            model.add_loss(loss)
            return model

        else:
            inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
            targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
            internal_model = Transformer(params, name="transformer_v2")
            ret = internal_model([inputs, targets], training=is_train)
            outputs = ret
            return tf.keras.Model([inputs, targets], outputs)


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
        self.encoder_stack = EncoderStack(params)
        self.decoder_stack = DecoderStack(params)

    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self, inputs, **kwargs):
        """Calculate target logits or inferred target sequences.

        Args:
          inputs: input tensor list of size 1 or 2.
            First item, inputs: int tensor with shape [batch_size, input_length].
            Second item (optional), targets: None or int tensor with shape
              [batch_size, target_length].
          training: boolean, whether in training mode or not.

        Returns:
          If targets is defined, then return logits for each word in the target
          sequence. float tensor with shape [batch_size, target_length, vocab_size]
          If target is none, then generate output sequence one token at a time.
            returns a dictionary {
              outputs: [batch_size, decoded length]
              scores: [batch_size, float]}
          Even when float16 is used, the output tensor(s) are always float32.

        Raises:
          NotImplementedError: If try to use padded decode method on CPU/GPUs.
        """
        source, targets = inputs[0], inputs[1]

        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("Transformer"):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = model_utils.get_padding_bias(source)

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(source, attention_bias, self.params['train'])
            # Generate output sequence if targets is None, or return logits if target
            # sequence is known.
            logits = self.decode(targets, encoder_outputs, attention_bias, self.params['train'])
            return logits

    def predict(self, inputs, **kwargs):
        source, targets = inputs[0], inputs[1]
        with tf.name_scope("Transformer_Predict"):
            attention_bias = model_utils.get_padding_bias(source)
            encoder_outputs = self.encode(source, attention_bias, self.params['train'])
            logits = self.decode(targets, encoder_outputs, attention_bias, self.params['train'])
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
        with tf.name_scope("encoder"):
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

            return self.encoder_stack(
                encoder_inputs, attention_bias, inputs_padding)

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
            decoder_inputs = tf.cast(decoder_inputs, self.params["dtype"])
            attention_bias = tf.cast(attention_bias, self.params["dtype"])
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(decoder_inputs,
                                        [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params["hidden_size"])
                pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
                decoder_inputs += pos_encoding
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=self.params["layer_postprocess_dropout"])

            # Run values
            # decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
            #     length, dtype=self.params["dtype"])
            decoder_self_attention_bias = tf.ones([1, 1, length, length], dtype=self.params["dtype"])
            logits = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias,
                training=training)
            batch_size = tf.shape(logits)[0]
            length = tf.shape(logits)[1]
            hidden_size = tf.shape(logits)[2]
            # logits = tf.reduce_mean(logits, axis=1)
            logits = tf.reshape(logits, [batch_size, length * hidden_size])
            logits = tf.reshape(logits, [batch_size, self.params['max_length'] * hidden_size])
            logits = self.embedding_layer(logits, mode="linear")

            # logits = tf.reshape(logits, [batch_size, 2])
            logits = tf.cast(logits, tf.float32)
            return logits


class LayerNormalization(tf.keras.layers.Layer):
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
            # layer_cache = cache[layer_name] if cache is not None else None
            with tf.compat.v1.variable_scope("decode_layer", reuse=tf.compat.v1.AUTO_REUSE):
                with tf.compat.v1.variable_scope("self_attention", reuse=tf.compat.v1.AUTO_REUSE):
                    decoder_inputs = self.self_attention_layer(
                        decoder_inputs,
                        decoder_self_attention_bias,
                        training=training)
                with tf.compat.v1.variable_scope("encdec_attention", reuse=tf.compat.v1.AUTO_REUSE):
                    decoder_inputs = self.enc_dec_attention_layer(
                        decoder_inputs,
                        encoder_outputs,
                        attention_bias,
                        training=training)
                with tf.name_scope("ffn"):
                    decoder_inputs = self.feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)
