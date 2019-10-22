# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-09-19 19:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def decode_record(record, name_to_features):
    example = tf.io.parse_single_example(record, name_to_features)
    return example


def input_fn_builder(file_path, name_to_features):
    def input_fn():
        dataset = tf.data.TFRecordDataset(file_path)
        dataset = dataset.map(lambda record: decode_record(record, name_to_features))
        if 'train' not in file_path:
            option = tf.data.Options()
            option.experimental_distribute.auto_shard = False
            dataset = dataset.with_options(option)
        return dataset

    return input_fn


def create_classifier_dataset(file_path, seq_length, batch_size, is_training=True, drop_remainder=True):
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'label_ids': tf.io.FixedLenFeature([], tf.int64),
    }
    input_fn = input_fn_builder(file_path, name_to_features)
    dataset = input_fn()

    def _select_data_from_record(record):
        x = {
            'input_ids': record['input_ids'],
        }
        y = record['label_ids']
        return x, y

    buffer_size = 24999
    dataset = dataset.map(_select_data_from_record)

    if is_training:
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    # dataset = dataset.prefetch(1024)
    return dataset

