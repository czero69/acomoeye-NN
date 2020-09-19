#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import random
import os
import argparse
import numpy as np
import tensorflow as tf
from et.trainer import GazeVecUtils, InferenceEngine, InputData
from tfutils.helpers import load_module
import sys
from matplotlib import pyplot as plt # for debug show purposes only
import tensorflow.contrib.slim as slim # for printing summary

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile_py.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def parse_args():
  parser = argparse.ArgumentParser(description='Perform training of a model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  parser.add_argument('--init_checkpoint', default=None, help='Path to checkpoint')
  return parser.parse_args()

# pylint: disable=too-many-locals, too-many-statements
def train(config, init_checkpoint):
  if hasattr(config.train, 'random_seed'):
    np.random.seed(config.train.random_seed)
    tf.set_random_seed(config.train.random_seed)
    random.seed(config.train.random_seed)

  if hasattr(config.train.execution, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train.execution.CUDA_VISIBLE_DEVICES

  input_train_data = InputData(batch_size=config.train.batch_size,
                               input_shape=config.input_shape,
                               file_list_path=config.train.file_list_path,
                               apply_basic_aug=config.train.apply_basic_aug,
                               apply_stn_aug=config.train.apply_stn_aug,
                               apply_coarse_dropout=config.train.apply_coarse_dropout,
                               apply_blur_aug=config.train.apply_blur_aug)


  graph = tf.Graph()

  inference_engine = InferenceEngine(first_layer_channels = config.first_layer_channels, do_coordConv = config.use_coord_conv, do_globalContext = config.use_global_context, do_fireBlocks = config.use_fireblocks, do_selfAttention = config.use_self_attention)

  with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    input_data, input_eyeID, input_gazeX, input_gazeY = input_train_data.input_fn()

    y_pred = inference_engine(input=input_data, num_gaze_vectors=config.num_gaze_vectors)
    shape = y_pred.get_shape()
    decoded_input = tf.numpy_function(GazeVecUtils.decode_gaze_vec, [input_gazeX, input_gazeY, config.num_gaze_vectors], tf.float32)
    decoded_input.set_shape(shape)

    gaze_vec_loss = tf.losses.huber_loss(labels=decoded_input, predictions=y_pred, delta=0.2) # smooth L1 loss
    loss = tf.reduce_mean(gaze_vec_loss)
    learning_rate = tf.train.piecewise_constant(global_step, [600000, 800000, 900000],
                                                [config.train.learning_rate, 0.1 * config.train.learning_rate,
                                                 0.01 * config.train.learning_rate, 0.001 * config.train.learning_rate])
    opt_loss = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, config.train.opt_type,
                                               config.train.grad_noise_scale, name='train_step')
    tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1000, write_version=tf.train.SaverDef.V2, save_relative_paths=True)

  conf = tf.ConfigProto()
  if hasattr(config.train.execution, 'per_process_gpu_memory_fraction'):
    conf.gpu_options.per_process_gpu_memory_fraction = config.train.execution.per_process_gpu_memory_fraction
  if hasattr(config.train.execution, 'allow_growth'):
    conf.gpu_options.allow_growth = config.train.execution.allow_growth

  session = tf.Session(graph=graph, config=conf)
  coordinator = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

  session.run('init')

  if init_checkpoint:
    tf.logging.info('Initialize from: ' + init_checkpoint)
    saver.restore(session, init_checkpoint)
  else:
    lastest_checkpoint = tf.train.latest_checkpoint(config.model_dir)
    if lastest_checkpoint:
      tf.logging.info('Restore from: ' + lastest_checkpoint)
      saver.restore(session, lastest_checkpoint)

  writer = None
  if config.train.need_to_save_log:
    writer = tf.summary.FileWriter(config.model_dir, session.graph)

  graph.finalize()

  for i in range(config.train.steps):
    if(config.train.debug_show_images):
        curr_step, curr_learning_rate, curr_loss, curr_opt_loss, curr_images = session.run([global_step, learning_rate, loss, opt_loss, input_data])
        print("len: ", len(curr_images))
        #plt.gray()
        for i in range(len(curr_images)):
            print("batch.i : ", i)
            plt.imshow(curr_images[i].astype(np.float32)[:, :, 0], cmap='gray', interpolation='nearest')
            plt.show()
    else:
        curr_step, curr_learning_rate, curr_loss, curr_opt_loss = session.run([global_step, learning_rate, loss, opt_loss])

    if i % config.train.display_iter == 0:
      if config.train.need_to_save_log:

        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='train/loss',
                                                              simple_value=float(curr_loss)),
                                             tf.Summary.Value(tag='train/learning_rate',
                                                              simple_value=float(curr_learning_rate)),
                                             tf.Summary.Value(tag='train/optimization_loss',
                                                              simple_value=float(curr_opt_loss))
                                             ]),
                           curr_step)
        writer.flush()

      if i % 500 == 0:
        print('Iteration: ' + str(curr_step) + ', Train loss: ' + str(curr_loss) + ', Learning rate: ' + str(curr_learning_rate))

    if ((curr_step % config.train.save_checkpoints_steps == 0 or curr_step == config.train.steps)
        and config.train.need_to_save_weights):
      saver.save(session, config.model_dir + '/model.ckpt-{:d}.ckpt'.format(curr_step))

  coordinator.request_stop()
  coordinator.join(threads)
  session.close()


def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  train(cfg, args.init_checkpoint)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
