# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf
import sys
sys.path.append('~/Desktop/Stanford/CS224n/MemeProject')
sys.path.append('ops')
import time

import configuration
import show_and_tell_model
from tensorflow.python.platform import tf_logging as logging


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "InceptionCkpt/inception_v3.ckpt",
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("loss_file", "",
                        "Losses+globalstep saved here")

tf.logging.set_verbosity(tf.logging.INFO)

def train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.
  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.
  Returns:
    The total loss and a boolean indicating whether or not to stop training.
  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  total_loss, np_global_step = sess.run([train_op, global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                   np_global_step, total_loss, time_elapsed)
      if np_global_step % 200 == 0 or np_global_step == 1:
	with open(FLAGS.loss_file,'a') as f:
           f.write("\n%d\t%.4lf" % (np_global_step, total_loss))
	      

  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop




def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.train_dir, "--train_dir is required"
  assert FLAGS.loss_file, "--loss_file is required"

  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = FLAGS.input_file_pattern
  model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
  training_config = configuration.TrainingConfig()

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    model = show_and_tell_model.ShowAndTellModel(
        model_config, mode="train", train_inception=FLAGS.train_inception)
    model.build()

    # Set up the learning rate.
    learning_rate_decay_fn = None
    if FLAGS.train_inception:
      learning_rate = tf.constant(training_config.train_inception_learning_rate)
    else:
      learning_rate = tf.constant(training_config.initial_learning_rate)
      if training_config.learning_rate_decay_factor > 0:
        num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                 model_config.batch_size)
        decay_steps = int(num_batches_per_epoch *
                          training_config.num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_rate, global_step):
          return tf.train.exponential_decay(
              learning_rate,
              global_step,
              decay_steps=decay_steps,
              decay_rate=training_config.learning_rate_decay_factor,
              staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=training_config.optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

  # Run training.
  tf.contrib.slim.learning.train(
      train_op,
      train_dir,
      train_step_fn=train_step,
      log_every_n_steps=FLAGS.log_every_n_steps,
      graph=g,
      global_step=model.global_step,
      number_of_steps=FLAGS.number_of_steps,
      init_fn=model.init_fn,
      saver=saver)


if __name__ == "__main__":
  tf.app.run()
