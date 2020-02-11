# -*- coding: utf-8 -*-
"""
 Created by zaber on 2019-09-14 22:
"""
import functools
import math

from absl import app
from absl import flags

import data_lib
import input_pipeline
import model_training_utils
import tensorflow as tf
from transformer import create_model, Config

# export PYTHONPATH=/data/model ;CUDA_VISIBLE_DEVICES='6,7' python run_classifier.py
flags.DEFINE_integer('epoch', 2, 'epoch for training.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_string('train_example_path', './raw_data/train', 'Path to train example for classifier.')
flags.DEFINE_string('eval_example_path', './raw_data/eval', 'Path to eval example for classifier.')
flags.DEFINE_string('train_output_path', './train_data/train_output.txt', 'Path to train output for classifier.')
flags.DEFINE_string('eval_output_path', './train_data/eval_output.txt', 'Path to eval output for classifier.')
flags.DEFINE_string('vocab_file', './train_data/vocab.txt', 'Path to vocabulary file.')
flags.DEFINE_string('model_dir', './checkpoint', 'checkpoint')
flags.DEFINE_integer(
    'steps_per_loop', 200,
    'Number of steps per graph-mode loop. ')
flags.DEFINE_float('adam_beta1', 0.9, 'optimizer_adam_beta1')
flags.DEFINE_float('adam_beta2', 0.997, 'optimizer_adam_beta2')
flags.DEFINE_float('optimizer_adam_epsilon', 1e-09, 'optimizer_adam_epsilon')
flags.DEFINE_float('learning_rate', 1e-3, 'The initial learning rate for Adam.')
flags.DEFINE_string('init_checkpoint', None, 'Initial checkpoint file')
flags.DEFINE_boolean('generate_dataset', False, 'generate dataset')
FLAGS = flags.FLAGS
config = Config()
params = dict()
params["batch_size"] = 32
params["hidden_size"] = 256
params["embedding_size"] = params["hidden_size"] / 8
params["max_seq_length"] = 256
params["vocab_size"] = config.vocab_size
params["train"] = True
params["dtype"] = tf.float32
params["layer_postprocess_dropout"] = 0.3
params["num_heads"] = 6
params["attention_dropout"] = 0.1
params["relu_dropout"] = 0.1
params["filter_size"] = 256
params["num_hidden_layers"] = 4


def generate_classifier_dataset():
    # data_lib.write_raw_data(FLAGS.train_example_path, FLAGS.eval_example_path)
    """Generates classifier dataset and returns input meta data."""
    return data_lib.generate_record_from_file(train_example_path=FLAGS.train_example_path,
                                              eval_example_path=FLAGS.eval_example_path,
                                              vocab_file=FLAGS.vocab_file,
                                              train_output_path=FLAGS.train_output_path,
                                              eval_output_path=FLAGS.eval_output_path,
                                              max_seq_length=params["max_seq_length"],
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
        core_model = create_model(params=params)
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
        "train_data_size": 3991585,
        "eval_data_size": 2000,
    }
    if FLAGS.generate_dataset:
        input_meta_data = generate_classifier_dataset()
    config.meta_data = input_meta_data

    run_customized_training()


if __name__ == '__main__':
    app.run(main)
