# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Translate text or files using trained transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from official.transformer.utils import tokenizer

_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6


def _get_sorted_inputs(source_file, target_file):
    with tf.io.gfile.GFile(source_file) as f:
        records = f.read().split("\n")
        source = [record.strip() for record in records]
        if not source[-1]:
            source.pop()
    with tf.io.gfile.GFile(target_file) as f:
        records = f.read().split("\n")
        target = [record.strip() for record in records]
        if not target[-1]:
            target.pop()

    sorted_keys = [i for i in range(len(target))]
    return source, target, sorted_keys


def _encode_and_add_eos(line, subtokenizer):
    """Encode line with subtokenizer, and add EOS id to the end."""
    return subtokenizer.encode(line) + [tokenizer.EOS_ID]


def _trim_and_decode(ids, subtokenizer):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    try:
        index = list(ids).index(tokenizer.EOS_ID)
        return subtokenizer.decode(ids[:index])
    except ValueError:  # No EOS found in sequence
        return subtokenizer.decode(ids)


def eval_file(model,
              params,
              subtokenizer,
              source_file,
              target_file,
              output_file=None,
              print_all_translations=True,
              distribution_strategy=None):
    batch_size = params["decode_batch_size"]

    # Read and sort inputs by length. Keep dictionary (original index-->new index
    # in sorted list) to write translations in the original order.
    source, target, sorted_keys = _get_sorted_inputs(source_file, target_file)
    total_samples = len(source)
    num_decode_batches = (total_samples - 1) // batch_size + 1

    def input_generator():
        """Yield encoded strings from sorted_inputs."""
        for i in range(num_decode_batches):
            source_line = [
                source[j + i * batch_size]
                for j in range(batch_size)
                if j + i * batch_size < total_samples
            ]
            target_line = [
                target[j + i * batch_size]
                for j in range(batch_size)
                if j + i * batch_size < total_samples
            ]
            source_line = [_encode_and_add_eos(l, subtokenizer) for l in source_line]
            target_line = [_encode_and_add_eos(l, subtokenizer) for l in target_line]
            if distribution_strategy:
                for j in range(batch_size - len(source_line)):
                    source_line.append([tokenizer.EOS_ID])
                    target_line.append([tokenizer.EOS_ID])
            source_batch = tf.keras.preprocessing.sequence.pad_sequences(
                source_line,
                maxlen=params["max_length"],
                dtype="int32",
                padding="post")
            target_batch = tf.keras.preprocessing.sequence.pad_sequences(
                source_line,
                maxlen=params["max_length"],
                dtype="int32",
                padding="post")
            tf.compat.v1.logging.info("Decoding batch %d out of %d.", i,
                                      num_decode_batches)
            yield source_batch, target_batch

    @tf.function
    def predict_step(inputs):
        """Decoding step function for TPU runs."""

        def _step_fn(inputs):
            """Per replica step function."""
            tag = inputs[0]
            val_inputs = inputs[1]
            val_outputs, _ = model([val_inputs], training=False)
            return tag, val_outputs

        return distribution_strategy.experimental_run_v2(_step_fn, args=(inputs,))

    results = []
    # if distribution_strategy:
    #     num_replicas = distribution_strategy.num_replicas_in_sync
    #     local_batch_size = params["decode_batch_size"] // num_replicas
    for i, (source_text, target_text) in enumerate(input_generator()):
        val_outputs = model.predict([source_text, target_text])
        length = len(val_outputs)
        for j in range(length):
            if j + i * batch_size < total_samples:
                res = val_outputs[j]
                results.append(res)
                # tf.compat.v1.logging.info(
                #         "Predicting:\n\tInput: %s\n\t%s\n\tOutput: %s" %
                #         (source[j + i * batch_size], target[j + i * batch_size], res))

    # Write translations in the order they appeared in the original file.
    if output_file is not None:
        if tf.io.gfile.isdir(output_file):
            raise ValueError("File output is a directory, will not save outputs to "
                             "file.")
        tf.compat.v1.logging.info("Writing to file %s" % output_file)
        with tf.compat.v1.gfile.Open(output_file, "w") as f:
            for i in sorted_keys:
                f.write("%s\n%s\n%s\n" % (source[i],target[i],results[i]))


def translate_from_text(model, subtokenizer, txt):
    encoded_txt = _encode_and_add_eos(txt, subtokenizer)
    result = model.predict(encoded_txt)
    outputs = result["outputs"]
    tf.compat.v1.logging.info("Original: \"%s\"" % txt)
    translate_from_input(outputs, subtokenizer)


def translate_from_input(outputs, subtokenizer):
    translation = _trim_and_decode(outputs, subtokenizer)
    tf.compat.v1.logging.info("Translation: \"%s\"" % translation)
