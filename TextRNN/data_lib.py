# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-09-20 14:
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import tokenizer
from absl import logging
import collections
import csv


class InputExample:
    def __init__(self, eid, text, label):
        self.eid = eid
        self.text = text
        self.label = label


class InputFeatures:
    def __init__(self, input_ids, label_id):
        self.input_ids = input_ids
        self.label_id = label_id


class DataProcessor:
    def get_train_examples(self, input_path):
        return self._create_examples(self._read_csv(input_path), "train")

    def get_dev_examples(self, input_path):
        return self._create_examples(self._read_csv(input_path), "dev")

    def get_test_examples(self, input_path):
        return self._create_examples(self._read_csv(input_path), "test")

    @staticmethod
    def get_labels():
        return ['0', '1']

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a tab separated value file."""
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            eid = "%s-%s" % (set_type, i)
            text = tokenizer.native_to_unicode(line[1])
            if set_type == "test":
                label = "0"
            else:
                label = tokenizer.native_to_unicode(line[0])
            examples.append(
                InputExample(eid=eid, text=text, label=label))
        return examples


def convert_example(ex_index, example, label_list, max_seq_length,
                    sub_tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_ids = sub_tokenizer.encode(example.text)

    # Zero-pad up to the sequence length.
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[0:max_seq_length]
    while len(input_ids) < max_seq_length:
        input_ids.append(0)

    assert len(input_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("rid: %s", example.eid)
        logging.info("tokens: %s", sub_tokenizer.decode(input_ids))
        logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logging.info("label: %s (id = %d)", example.label, label_id)

    feature = InputFeatures(
        input_ids=input_ids,
        label_id=label_id)
    return feature


def convert_examples_to_features(examples, label_list,
                                 max_seq_length, sub_tokenizer,
                                 output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.io.TFRecordWriter(output_file)

    for (i, e) in enumerate(examples):
        logging.info("Writing example %d of %d", i, len(examples))

        feature = convert_example(i, e, label_list, max_seq_length, sub_tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


# If set, use binary search to find the vocabulary set with size closest to the target size
def generate_record_from_file(train_example_path, eval_example_path, vocab_file, train_output_path,
                              eval_output_path, max_seq_length, config):
    processor = DataProcessor()
    label_list = processor.get_labels()
    sub_tokenizer = tokenizer.Subtokenizer.init_from_files(
        vocab_file, [train_example_path, eval_example_path], config.vocab_size, config.generate_threshold,
        min_count=None if config.generate_search else config.generate_DATA_MIN_COUNT)
    train_example = processor.get_train_examples(train_example_path)
    num_train_data = len(train_example)
    convert_examples_to_features(train_example, label_list, max_seq_length, sub_tokenizer, train_output_path)
    eval_example = processor.get_dev_examples(eval_example_path)
    convert_examples_to_features(eval_example, label_list, max_seq_length, sub_tokenizer, eval_output_path)
    num_eval_data = len(train_example)
    meta_data = {
        "num_labels": len(processor.get_labels()),
        "train_data_size": num_train_data,
        "eval_data_size": num_eval_data,
        "max_seq_length": max_seq_length,
    }
    return meta_data


def write_raw_data(train_example_path, eval_example_path):
    logging.info('Loading data...')

    def write(ds, ep):
        ds = tfds.as_numpy(
            ds,
            graph=None
        )
        example = []
        for t, l in ds:
            e = str(l) + '\t' + t.decode("utf-8") + '\n'
            example.append(e)
        with tf.io.gfile.GFile(ep, 'w') as f:
            for r in example:
                f.write(r)
        logging.info('%d sequences', len(example))

    train_dataset = tfds.load(name="imdb_reviews", split=tfds.Split.TRAIN, as_supervised=True)
    write(train_dataset, train_example_path)
    test_dataset = tfds.load(name="imdb_reviews", split=tfds.Split.TEST, as_supervised=True)
    write(test_dataset, eval_example_path)
