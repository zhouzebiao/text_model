from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import logging
import tensorflow as tf


_SUMMARY_TXT = 'training_summary.txt'
_MIN_SUMMARY_STEPS = 10


def _save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
    """Saves model to with provided checkpoint prefix."""

    checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
    saved_path = checkpoint.save(checkpoint_path)
    logging.info('Saving model as TF checkpoint: %s', saved_path)
    return


def _get_input_iterator(input_fn, strategy):
    input_data = input_fn()
    if callable(input_data):
        iterator = iter(
            strategy.experimental_distribute_datasets_from_function(input_data))
    else:
        iterator = iter(strategy.experimental_distribute_dataset(input_data))
    return iterator


def _float_metric_value(metric):
    """Gets the value of a float-value keras metric."""
    return metric.result().numpy().astype(float)


def _steps_to_run(current_step, steps_per_epoch, steps_per_loop):
    """Calculates steps to run on device."""
    if steps_per_loop <= 0:
        raise ValueError('steps_per_loop should be positive integer.')
    if steps_per_loop == 1:
        return steps_per_loop
    remainder_in_epoch = current_step % steps_per_epoch
    if remainder_in_epoch != 0:
        return min(steps_per_epoch - remainder_in_epoch, steps_per_loop)
    else:
        return steps_per_loop


def _write_txt_summary(training_summary, model_dir):
    """Writes a summary text file to record stats."""
    summary_path = os.path.join(model_dir, _SUMMARY_TXT)
    with tf.io.gfile.GFile(summary_path, 'wb') as f:
        logging.info('Training Summary: \n%s', str(training_summary))
        f.write(json.dumps(training_summary, indent=4))


def run_customized_training_loop(
        strategy=None,
        model_fn=None,
        loss_fn=None,
        model_dir=None,
        train_input_fn=None,
        steps_per_epoch=None,
        steps_per_loop=1,
        epochs=1,
        eval_input_fn=None,
        eval_steps=None,
        init_checkpoint=None,
        custom_callbacks=None):

    total_training_steps = steps_per_epoch * epochs

    # To reduce unnecessary send/receive input pipeline operation, we place input
    # pipeline ops in worker task.
    train_iterator = _get_input_iterator(train_input_fn, strategy)
    eval_iterator = _get_input_iterator(eval_input_fn, strategy)

    with strategy.scope():
        # To correctly place the model weights on accelerators,
        # model and optimizer should be created in scope.
        model = model_fn()
        if not hasattr(model, 'optimizer'):
            raise ValueError('User should set optimizer attribute to model '
                             'inside `model_fn`.')
        optimizer = model.optimizer
        # use_float16 = isinstance(
        #     optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer)

        if init_checkpoint:
            logging.info(
                'Checkpoint file %s found and restoring from '
                'initial checkpoint for core model.', init_checkpoint)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(init_checkpoint).assert_consumed()
            logging.info('Loading from checkpoint file completed')

        train_loss_metric = tf.keras.metrics.Mean(
            'training_loss', dtype=tf.float32)
        eval_loss_metric = tf.keras.metrics.Mean(name='eval_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='eval_accuracy')
        # Create summary writers
        eval_summary_writer = tf.summary.create_file_writer(
            os.path.join(model_dir, 'summaries/eval'))
        if steps_per_loop >= _MIN_SUMMARY_STEPS:
            # Only writes summary when the stats are collected sufficiently over
            # enough steps.
            train_summary_writer = tf.summary.create_file_writer(
                os.path.join(model_dir, 'summaries/train'))
        else:
            train_summary_writer = None

        def train_step(inputs):
            """Replicated training step."""

            input_ids, labels = inputs[0]['input_ids'], inputs[1]
            with tf.GradientTape() as tape:
                predictions = model(input_ids, training=True)
                loss = loss_fn(labels, predictions)
                # if use_float16:
                #     scaled_loss = optimizer.get_scaled_loss(loss)

            # if use_float16:
            #     scaled_grads = tape.gradient(scaled_loss, training_vars)
            #     grads = optimizer.get_unscaled_gradients(scaled_grads)
            # else:
            #     grads = tape.gradient(loss, training_vars)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # For reporting, the metric takes the mean of losses.
            train_loss_metric.update_state(loss)
            train_accuracy.update_state(labels, predictions)
            return loss

        def eval_step(inputs):
            input_ids, labels = inputs[0]['input_ids'], inputs[1]

            predictions = model(input_ids, training=False)
            e_loss = loss_fn(labels, predictions)

            eval_loss_metric.update_state(e_loss)
            eval_accuracy.update_state(labels, predictions)

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.experimental_run_v2(train_step,
                                                              args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)

        @tf.function
        def distributed_eval_step(dataset_inputs):
            return strategy.experimental_run_v2(eval_step, args=(dataset_inputs,))

        def _run_evaluation(current_training_step, iterator):
            """Runs validation steps and aggregate metrics."""
            distributed_eval_step(next(iterator))
            eval_template = "Eval Step: %d   Test Loss: %f, Test Accuracy: %f" \
                            % (eval_steps,
                               _float_metric_value(eval_loss_metric),
                               _float_metric_value(eval_accuracy) * 100)
            logging.info(eval_template)
            with eval_summary_writer.as_default():
                tf.summary.scalar(
                    eval_loss_metric.name, _float_metric_value(eval_loss_metric), step=current_training_step)
                tf.summary.scalar(
                    eval_accuracy.name, _float_metric_value(eval_accuracy), step=current_training_step)

                eval_summary_writer.flush()

        def _run_callbacks_on_batch_begin(batch):
            """Runs custom callbacks at the start of every step."""
            if not custom_callbacks:
                return
            for callback in custom_callbacks:
                callback.on_batch_begin(batch)

        def _run_callbacks_on_batch_end(batch):
            """Runs custom callbacks at the end of every step."""
            if not custom_callbacks:
                return
            for callback in custom_callbacks:
                callback.on_batch_end(batch)

        # Training loop starts here.
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint_file:
            logging.info(
                'Checkpoint file %s found and restoring from '
                'checkpoint', latest_checkpoint_file)
            checkpoint.restore(latest_checkpoint_file)
            logging.info('Loading from checkpoint file completed')

        current_step = optimizer.iterations.numpy()
        checkpoint_name = 'ctl_step_{step}.ckpt'

        while current_step < total_training_steps:
            # Training loss/metric are taking average over steps inside micro
            # training loop. We reset the their values before each round.
            train_loss_metric.reset_states()
            eval_loss_metric.reset_states()
            train_accuracy.reset_states()
            eval_accuracy.reset_states()
            for metric in model.metrics:
                metric.reset_states()

            _run_callbacks_on_batch_begin(current_step)
            # Runs several steps in the host while loop.
            steps = _steps_to_run(current_step, steps_per_epoch, steps_per_loop)
            total_loss = 0.0
            for i in range(steps):
                total_loss += distributed_train_step(next(train_iterator))
            train_loss = total_loss / steps

            _run_callbacks_on_batch_end(current_step)
            current_step += steps

            # train_loss = _float_metric_value(train_loss_metric)
            # Updates training logging.
            template = "Epoch %d, Train Step: %d/%d  Loss: %f, Accuracy: %f" % (
                current_step / steps_per_epoch, current_step, total_training_steps,
                train_loss, _float_metric_value(train_accuracy) * 100)

            if train_summary_writer:
                with train_summary_writer.as_default():
                    tf.summary.scalar(
                        train_loss_metric.name, train_loss, step=current_step)
                    tf.summary.scalar(
                        train_accuracy.name, _float_metric_value(train_accuracy), step=current_step)
                    train_summary_writer.flush()
            logging.info(template)

            # Saves model checkpoints and run validation steps at every epoch end.
            if current_step % steps_per_epoch == 0:
                # To avoid repeated model saving, we do not save after the last
                # step of training.
                if current_step < total_training_steps:
                    _save_checkpoint(checkpoint, model_dir,
                                     checkpoint_name.format(step=current_step))

                if eval_input_fn:
                    logging.info('Running evaluation after step: %s.', current_step)
                    _run_evaluation(current_step, eval_iterator)
                    # Re-initialize evaluation metric.
                    for metric in model.metrics:
                        metric.reset_states()

        _save_checkpoint(checkpoint, model_dir,
                         checkpoint_name.format(step=current_step))

        if eval_input_fn:
            logging.info('Running final evaluation after training is complete.')
            _run_evaluation(current_step, eval_iterator)

        training_summary = {
            'total_training_steps': total_training_steps,
            'train_loss': _float_metric_value(train_loss_metric),
        }

        _write_txt_summary(training_summary, model_dir)

        return model
