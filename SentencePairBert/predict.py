# -*- coding: utf-8 -*-
"""
 Created by zaber on 2020-02-28 03:
"""

import json

import numpy as np
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
from official.nlp import bert_modeling as modeling
from official.nlp import bert_models
from official.nlp.bert import common_flags
from official.nlp.bert import input_pipeline

# pylint: disable=unused-import,g-import-not-at-top,redefined-outer-name,reimported

flags.DEFINE_integer('eval_batch_size', 218, 'Batch size for evaluation.')
flags.DEFINE_string('eval_data_path', None,
                    'Path to evaluation data for BERT classifier.')
flags.DEFINE_string('vocab_file', None,
                    'The vocabulary file that the BERT model was trained on.')
flags.DEFINE_bool(
    'do_lower_case', True,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def get_raw_results(predictions):
    """Converts multi-replica predictions to RawResult."""
    for logits, label_ids, mask in zip(predictions['logits'], predictions['label_ids'], predictions['mask']):
        for lo, la, ma in zip(logits.numpy(), label_ids.numpy(), mask.numpy()):
            yield {'logits': lo, 'label_ids': la, 'mask': ma, }


def predict_customized(strategy, input_meta_data, bert_config,
                       eval_data_path, num_steps):
    max_seq_length = input_meta_data['max_seq_length']
    num_classes = input_meta_data['num_labels']
    predict_dataset = input_pipeline.create_classifier_dataset(
        eval_data_path,
        input_meta_data['max_seq_length'],
        FLAGS.eval_batch_size,
        is_training=False)
    predict_iterator = iter(
        strategy.experimental_distribute_dataset(predict_dataset))
    with strategy.scope():
        tf.keras.mixed_precision.experimental.set_policy('float32')
        classifier_model, _ = (
            bert_models.classifier_model(
                bert_config,
                tf.float32,
                num_classes,
                max_seq_length))
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    logging.info('Restoring checkpoints from %s', checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=classifier_model)
    checkpoint.restore(checkpoint_path).expect_partial()

    @tf.function
    def predict_step(iterator):
        """Predicts on distributed devices."""

        def _replicated_step(inputs_d):
            """Replicated prediction calculation."""
            inputs, label = inputs_d
            # x = {
            #     'input_word_ids': inputs['input_word_ids'],
            #     'input_mask': inputs['input_mask'],
            #     'input_type_ids': inputs['input_type_ids'],
            # }
            logits = classifier_model(inputs, training=False)
            return dict(logits=logits, label_ids=label, mask=inputs["is_real_example"])

        outputs = strategy.experimental_run_v2(
            _replicated_step, args=(next(iterator),))
        return tf.nest.map_structure(strategy.experimental_local_results, outputs)

    correct = 0
    total = 0
    all_results = []
    for _ in range(num_steps):
        predictions = predict_step(predict_iterator)
        merged_logits = []
        merged_labels = []
        merged_masks = []
        for result in get_raw_results(predictions):
            all_results.append(result)
        if len(all_results) % 100 == 0:
            logging.info('Made predictions for %d records.', len(all_results))
        for logits, label_ids, mask in zip(predictions['logits'], predictions['label_ids'], predictions['mask']):
            merged_logits.append(logits)
            merged_labels.append(label_ids)
            merged_masks.append(mask)
        merged_logits = np.vstack(np.array(merged_logits))
        merged_labels = np.hstack(np.array(merged_labels))
        merged_masks = np.hstack(np.array(merged_masks))
        real_index = np.where(np.equal(merged_masks, 1))
        correct += np.sum(
            np.equal(
                np.argmax(merged_logits, axis=-1),
                merged_labels))
        total += np.shape(real_index)[-1]
    accuracy = float(correct) / float(total)
    logging.info("Train step: %d  /  acc = %d/%d = %f", num_steps, correct, total,
                 accuracy)
    return all_results


def predict_classifier(strategy, input_meta_data):
    """Makes predictions for a squad dataset."""
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    dataset_size = 1999
    num_steps = int(dataset_size / FLAGS.eval_batch_size)
    all_results = predict_customized(strategy, input_meta_data, bert_config, FLAGS.eval_data_path
                                     , num_steps)
    for r in all_results:
        pass


def main(_):
    # Users should always run this script under TF 2.x
    assert tf.version.VERSION.startswith('2.')

    with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
        input_meta_data = json.loads(reader.read().decode('utf-8'))

    if not FLAGS.model_dir:
        FLAGS.model_dir = '/tmp/bert20/'

    strategy = tf.distribute.MirroredStrategy()

    predict_classifier(strategy, input_meta_data)


if __name__ == '__main__':
    flags.mark_flag_as_required('bert_config_file')
    app.run(main)
