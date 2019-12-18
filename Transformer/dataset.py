# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-12-16 15:
"""
import os

import tensorflow as tf

_READ_RECORD_BUFFER = 8 * 1000 * 1000

_MIN_BOUNDARY = 4
_BOUNDARY_SCALE = 2


def _batch_examples(dataset, batch_size, max_length):
    def create_min_max_boundaries(
            length, min_boundary=_MIN_BOUNDARY, boundary_scale=_BOUNDARY_SCALE):
        bucket_boundaries = [int(min_boundary)]
        x = min_boundary
        while x < length:
            x = max(bucket_boundaries[-1] + 1, int(bucket_boundaries[-1] * boundary_scale))
            x = min(x, length)
            bucket_boundaries.append(x)

        bucket_min = [0] + bucket_boundaries
        bucket_max = bucket_boundaries + [length + 1]
        return bucket_min, bucket_max

    buckets_min, buckets_max = create_min_max_boundaries(max_length)

    # 根据子词长度决定batch size 大小
    bucket_batch_sizes = tf.constant([batch_size // x for x in buckets_max], dtype=tf.int64)

    def example_to_bucket_id(example_source, example_target):
        seq_length = tf.maximum(tf.shape(example_source)[0], tf.shape(example_target)[0])
        # 两条件为true，判定属于哪个分桶
        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length),
            tf.less(seq_length, buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        return grouped_dataset.padded_batch(window_size_fn(bucket_id), ([None], [None]))

    return dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn, window_size=None,
        window_size_func=window_size_fn))


def _read_and_batch_from_files(
        file_pattern, batch_size, max_length, num_parallel_calls, shuffle, repeat):
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    def load_records(filename):
        return tf.data.TFRecordDataset(filename, buffer_size=_READ_RECORD_BUFFER)

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            load_records, sloppy=shuffle, cycle_length=num_parallel_calls))

    def _parse_example(serialized_example):
        data_fields = {
            "source": tf.io.VarLenFeature(tf.int64),
            "target": tf.io.VarLenFeature(tf.int64)
        }
        parsed = tf.io.parse_single_example(serialized_example, data_fields)
        return tf.sparse.to_dense(parsed["source"]), tf.sparse.to_dense(parsed["target"])

    dataset = dataset.map(_parse_example,
                          num_parallel_calls=num_parallel_calls)

    def _filter_max_length(example, length=256):
        return tf.logical_and(tf.size(example[0]) <= length,
                              tf.size(example[1]) <= length)

    dataset = dataset.filter(lambda x, y: _filter_max_length((x, y), max_length))

    dataset = _batch_examples(dataset, batch_size, max_length)

    dataset = dataset.repeat(repeat)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def train_input_fn(params):
    file_pattern = os.path.join(params["data_dir"] or "", "*train*")
    return _read_and_batch_from_files(
        file_pattern, params["batch_size"], params["max_length"],
        params["num_parallel_calls"], shuffle=True,
        repeat=params["repeat_dataset"])


def eval_input_fn(params):
    file_pattern = os.path.join(params["data_dir"] or "", "*dev*")
    return _read_and_batch_from_files(
        file_pattern, params["batch_size"], params["max_length"],
        params["num_parallel_calls"], shuffle=False, repeat=1)
