# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-09-14 22:
"""
import functools
from absl import flags
from absl import logging
from absl import app
import data_lib
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from text_rnn import get_model, Config
import input_pipeline
import tensorflow as tf
import model_training_utils
import math
# CUDA_VISIBLE_DEVICES='6,7' python run_classifier.py
flags.DEFINE_integer('epoch', 10, 'epoch for training.')
flags.DEFINE_integer('batch_size', 512, 'Batch size for training.')
flags.DEFINE_string('train_example_path', './raw_data/train_example.txt', 'Path to train example for classifier.')
flags.DEFINE_string('eval_example_path', './raw_data/eval_example.txt', 'Path to eval example for classifier.')
flags.DEFINE_string('train_output_path', './train_data/train_output.txt', 'Path to train output for classifier.')
flags.DEFINE_string('eval_output_path', './train_data/eval_output.txt', 'Path to eval output for classifier.')
flags.DEFINE_string('vocab_file', './train_data/vocab.txt', 'Path to vocabulary file.')
flags.DEFINE_string('model_dir', './checkpoint', 'checkpoint')
flags.DEFINE_integer(
    'steps_per_loop', 20,
    'Number of steps per graph-mode loop. ')
flags.DEFINE_float('adam_beta1', 0.9, 'optimizer_adam_beta1')
flags.DEFINE_float('adam_beta2', 0.997, 'optimizer_adam_beta2')
flags.DEFINE_float('optimizer_adam_epsilon', 1e-09, 'optimizer_adam_epsilon')
flags.DEFINE_float('learning_rate', 1e-3, 'The initial learning rate for Adam.')
flags.DEFINE_string('init_checkpoint', None, 'Initial checkpoint file')
flags.DEFINE_boolean('generate_dataset', False, 'generate dataset')
FLAGS = flags.FLAGS
config = Config()


def generate_classifier_dataset():
    data_lib.write_raw_data(FLAGS.train_example_path, FLAGS.eval_example_path)
    """Generates classifier dataset and returns input meta data."""
    return data_lib.generate_record_from_file(train_example_path=FLAGS.train_example_path,
                                              eval_example_path=FLAGS.eval_example_path,
                                              vocab_file=FLAGS.vocab_file,
                                              train_output_path=FLAGS.train_output_path,
                                              eval_output_path=FLAGS.eval_output_path,
                                              max_seq_length=config.max_seq_length,
                                              config=config)


def get_loss_fn():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)

    def loss_fn(labels, logits):
        per_example_loss = loss_object(labels, logits)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=FLAGS.batch_size)

    return loss_fn


def run_customized_training(custom_callbacks=None):
    train_data_size = config.meta_data['train_data_size']
    eval_data_size = config.meta_data['eval_data_size']
    strategy = tf.distribute.MirroredStrategy()
    steps_per_epoch = int(train_data_size / FLAGS.batch_size / strategy.num_replicas_in_sync)
    eval_steps = int(
        math.ceil(eval_data_size / FLAGS.batch_size))
    global_batch_size = FLAGS.batch_size * strategy.num_replicas_in_sync
    train_input_fn = functools.partial(
        input_pipeline.create_classifier_dataset,
        file_path=FLAGS.train_output_path,
        seq_length=config.max_seq_length,
        batch_size=global_batch_size,
        buffer_size=train_data_size,
    )
    eval_input_fn = functools.partial(
        input_pipeline.create_classifier_dataset,
        file_path=FLAGS.eval_output_path,
        seq_length=config.max_seq_length,
        batch_size=global_batch_size,
        buffer_size=eval_data_size,
        is_training=False,
        drop_remainder=False)

    def _get_classifier_model():
        core_model = get_model(config=config,float_type=tf.float32)
        core_model.optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate,
                                                        beta_1=FLAGS.adam_beta1,
                                                        beta_2=FLAGS.adam_beta2,
                                                        epsilon=FLAGS.optimizer_adam_epsilon,
                                                        )
        return core_model

    loss_fn = get_loss_fn()

    return model_training_utils.run_customized_training_loop(
        strategy=strategy,
        model_fn=_get_classifier_model,
        loss_fn=loss_fn,
        model_dir=FLAGS.model_dir,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=FLAGS.steps_per_loop,
        epochs=FLAGS.epoch,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        eval_steps=eval_steps,
        init_checkpoint=FLAGS.init_checkpoint,
        custom_callbacks=custom_callbacks, )


def main(_):
    input_meta_data = {
        "train_data_size": 24999,
        "eval_data_size": 24999,
    }
    if FLAGS.generate_dataset:
        input_meta_data = generate_classifier_dataset()
    config.meta_data = input_meta_data

    run_customized_training()


if __name__ == '__main__':
    app.run(main)
