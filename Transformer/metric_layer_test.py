# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-11-26 17:
"""

import metrics
import tensorflow as tf


class MetricLayerTest(tf.test.TestCase):

    def test_metric_layer(self):
        vocab_size = 50
        data_type = tf.float16
        logits = tf.keras.layers.Input((None, vocab_size),
                                       dtype="float32",
                                       name="logits")
        targets = tf.keras.layers.Input((None,), dtype="int64", name="labels")
        output_logits = metrics.MetricLayer(vocab_size, data_type)([logits, targets])
        self.assertEqual(output_logits.shape.as_list(), [None, None, vocab_size, ])


if __name__ == "__main__":
    tf.test.main()
"""
export PYTHONPATH=/data/model;CUDA_VISIBLE_DEVICES='-1' python metric_layer_test.py

"""
