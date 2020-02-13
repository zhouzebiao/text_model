from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from absl import logging

import tensorflow as tf
from official.transformer.v2 import misc
# import model_helpers

# Buffer size for reading records from a TFRecord file. Each training file is
# 7.2 MB, so 8 MB allows an entire file to be kept in memory.
_READ_RECORD_BUFFER = 8 * 1000 * 1000

# Example grouping constants. Defines length boundaries for each group.
# These values are the defaults used in Tensor2Tensor.
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


def _load_records(filename):
    """Read file and return a dataset of tf.Examples."""
    return tf.data.TFRecordDataset(filename, buffer_size=_READ_RECORD_BUFFER)


def _parse_example(serialized_example):
    """Return inputs and targets Tensors from a serialized tf.Example."""
    data_fields = {
        "inputs": tf.io.VarLenFeature(tf.int64),
        "targets": tf.io.VarLenFeature(tf.int64),
        "labels": tf.io.VarLenFeature(tf.int64)
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    inputs = tf.sparse.to_dense(parsed["inputs"])
    targets = tf.sparse.to_dense(parsed["targets"])
    labels = tf.sparse.to_dense(parsed["labels"])
    return inputs, targets, labels


def _filter_max_length(example, max_length=256):
    """Indicates whether the example's length is lower than the maximum length."""
    return tf.logical_and(tf.size(example[0]) <= max_length,
                          tf.size(example[1]) <= max_length)


def _get_example_length(example):
    """Returns the maximum length between the example inputs and targets."""
    length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])
    return length


def _create_min_max_boundaries(
        max_length, min_boundary=_MIN_BOUNDARY, boundary_scale=_BOUNDARY_SCALE):
    """Create min and max boundary lists up to max_length.

    For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
    returned values will be:
      buckets_min = [0, 4, 8, 16, 24]
      buckets_max = [4, 8, 16, 24, 25]

    Args:
      max_length: The maximum length of example in dataset.
      min_boundary: Minimum length in boundary.
      boundary_scale: Amount to scale consecutive boundaries in the list.

    Returns:
      min and max boundary lists

    """
    # Create bucket boundaries list by scaling the previous boundary or adding 1
    # (to ensure increasing boundary sizes).
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))

    # Create min and max boundary lists from the initial list.
    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


def _batch_examples(dataset, batch_size, max_length):
    # Get min and max boundary lists for each example. These are used to calculate
    # the `bucket_id`, which is the index at which:
    # buckets_min[bucket_id] <= len(example) < buckets_max[bucket_id]
    # Note that using both min and max lists improves the performance.
    buckets_min, buckets_max = _create_min_max_boundaries(max_length)

    # Create list of batch sizes for each bucket_id, so that
    # bucket_batch_size[bucket_id] * buckets_max[bucket_id] <= batch_size
    bucket_batch_sizes = [batch_size // x for x in buckets_max]
    # bucket_id will be a tensor, so convert this list to a tensor as well.
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(example_input, example_target, example_label):
        """Return int64 bucket id for this example, calculated based on length."""
        seq_length = _get_example_length((example_input, example_target))

        # TODO(xunkai): investigate if removing code branching improves performance.
        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length),
            tf.less(seq_length, buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        """Return number of examples to be grouped when given a bucket id."""
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        """Batch and add padding to a dataset of elements with similar lengths."""
        bucket_batch_size = window_size_fn(bucket_id)

        # Batch the dataset and add padding so that all input sequences in the
        # examples have the same length, and all target sequences have the same
        # lengths as well. Resulting lengths of inputs and targets can differ.
        return grouped_dataset.padded_batch(bucket_batch_size, ([None], [None], [None]))

    return dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=None,
        window_size_func=window_size_fn))


def _read_and_batch_from_files(
        file_pattern, batch_size, max_length, num_parallel_calls, shuffle, repeat,
        static_batch=False, num_replicas=1, ctx=None):
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    if ctx and ctx.num_input_pipelines > 1:
        logging.info("Shard %d of the dataset.", ctx.input_pipeline_id)
        dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)

    # Read files and interleave results. When training, the order of the examples
    # will be non-deterministic.
    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.interleave(
        _load_records,
        cycle_length=num_parallel_calls,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)

    # Parse each tf.Example into a dictionary
    # TODO: Look into prefetch_input_elements for performance optimization.
    dataset = dataset.map(_parse_example,
                          num_parallel_calls=num_parallel_calls)

    # Remove examples where the input or target length exceeds the maximum length,
    dataset = dataset.filter(lambda x, y, z: _filter_max_length((x, y), max_length))

    if static_batch:
        dataset = dataset.padded_batch(
            # First calculate batch size (token number) per worker, then divide it
            # into sentences, and finally expand to a global batch. It could prove
            # the global batch divisble for distribution strategy.
            int(batch_size // num_replicas // max_length * num_replicas),
            ([max_length], [max_length]), drop_remainder=True)
    else:
        # Group and batch such that each batch has examples of similar length.
        # TODO(xunkai): _batch_examples might need to do something special for
        # num_replicas.
        dataset = _batch_examples(dataset, batch_size, max_length)

    dataset = dataset.repeat(repeat)

    # Prefetch the next element to improve speed of input pipeline.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


# def _generate_synthetic_data(params):
#     """Create synthetic data based on the parameter batch size."""
#     batch = length = int(math.sqrt(params["batch_size"]))
#     dataset = model_helpers.generate_synthetic_data(
#         source_shape=tf.TensorShape([length]),
#         source_value=1,
#         source_dtype=tf.int64,
#         target_shape=tf.TensorShape([length]),
#         target_value=1,
#         target_dtype=tf.int64,
#         label_shape=tf.TensorShape([length]),
#         label_value=1,
#         label_dtype=tf.int64,
#     )
#     return dataset.batch(batch, drop_remainder=True)


def train_input_fn(params, ctx=None):
    """Load and return dataset of batched examples for use during training."""
    file_pattern = os.path.join(params["data_dir"] or "", "*train*")
    return _read_and_batch_from_files(
        file_pattern, params["batch_size"], params["max_length"],
        params["num_parallel_calls"], shuffle=True,
        repeat=params["repeat_dataset"], static_batch=params["static_batch"],
        num_replicas=params["num_gpus"], ctx=ctx)


def eval_input_fn(params, ctx=None):
    """Load and return dataset of batched examples for use during evaluation."""
    file_pattern = os.path.join(params["data_dir"] or "", "*dev*")
    print(params["use_synthetic_data"], file_pattern, params["batch_size"], params["max_length"],
          params["num_parallel_calls"], params["static_batch"], params["num_gpus"], ctx)
    # if params["use_synthetic_data"]:
    #     return _generate_synthetic_data(params)
    return _read_and_batch_from_files(
        file_pattern, params["batch_size"], params["max_length"],
        params["num_parallel_calls"], shuffle=False, repeat=1,
        static_batch=params["static_batch"], num_replicas=params["num_gpus"],
        ctx=ctx)


def map_data_for_transformer_fn(x, y, z):
    """Maps data for training, and handles weried behaviors for different vers."""
    # Will transform input x and targets y into tuple(x, y) as new model inputs.
    if misc.is_v2():
        # For TF v2, the 2nd parameter is omitted to make Keras training work.
        return ((x, y, z),)
    else:
        # For TF v1, Keras requires a dummy placeholder as the 2nd parameter.
        return (x, y, z), tf.constant(0.0)
