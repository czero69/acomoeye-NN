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

from __future__ import print_function
import argparse
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from et.trainer import InferenceEngine, GazeVecUtils, augment, augment_with_stn
from et.utils import accuracy_projection, accuracy_yaw_pitch, dataset_size
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

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

sys.stdout = Logger()

def log_histogram(writer, tag, values, step, bins=100):
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
    writer.flush()

def parse_args():
  parser = argparse.ArgumentParser(description='Perform evaluation of a trained model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()

def read_data(height, width, channels_num, list_file_name, base_path, batch_size=10):
  reader = tf.TextLineReader()
  _, value = reader.read(list_file_name)
  #filename, label, country_code = tf.decode_csv(value, [[''], [''], ['']], ' ')
  filename, eyeID, gazeX, gazeY = tf.decode_csv(value, [[''], [''], [''], ['']], ',')
  image_file = tf.read_file(base_path + '/' + filename)

  # rgb_image = tf.image.decode_png(image_file, channels=channels_num)
  rgb_image = tf.image.decode_jpeg(image_file, channels=channels_num)
  rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
  resized_image = tf.image.resize_images(rgb_image_float, [height, width])
  resized_image.set_shape([height, width, channels_num])

  image_batch, eyeID_batch, gazeX_batch, gazeY_batch, file_batch, filename_batch = tf.train.batch([resized_image, eyeID, gazeX, gazeY, image_file, filename], batch_size=batch_size)
  return image_batch, eyeID_batch, gazeX_batch, gazeY_batch, file_batch, filename_batch


def data_input(height, width, channels_num, filename, base_path, batch_size=1):
  files_string_producer = tf.train.string_input_producer([filename])
  image, eyeID_label, gazeX_label, gazeY_label, file_batch, file_name = read_data(height, width, channels_num, files_string_producer, base_path, batch_size)

  #image = augment(image)
  #image = tf.cond(tf.constant(True, dtype=tf.bool), lambda: image, lambda: augment_with_stn(image))

  return image, eyeID_label, gazeX_label, gazeY_label, file_batch, file_name

# pylint: disable=too-many-branches, too-many-statements, too-many-locals
def validate(config):
  if hasattr(config.eval, 'random_seed'):
    np.random.seed(config.eval.random_seed)
    tf.set_random_seed(config.eval.random_seed)
    random.seed(config.eval.random_seed)

  if hasattr(config.eval.execution, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.eval.execution.CUDA_VISIBLE_DEVICES

  height, width, channels_num = config.input_shape

  if(config.accuracy_measure == "acomo"):
    accuracy = accuracy_yaw_pitch
  if(config.accuracy_measure == "nvgaze"):
    accuracy = accuracy_projection

  graph = tf.Graph()

  inference_engine = InferenceEngine(first_layer_channels = config.first_layer_channels, do_coordConv = config.use_coord_conv, do_globalContext = config.use_global_context, do_fireBlocks = config.use_fireblocks, do_selfAttention = config.use_self_attention)

  with graph.as_default():
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
      input_data, eyeID_label, gazeX_label, gazeY_label, file_batch, file_name = data_input(height, width, channels_num,
                                                   config.eval.file_list_path, os.path.dirname(config.eval.file_list_path), batch_size=config.eval.batch_size)

      y_pred = inference_engine(input=input_data, num_gaze_vectors=config.num_gaze_vectors)

      shape = y_pred.get_shape()
      decoded_gaze_input = tf.numpy_function(GazeVecUtils.decode_gaze_vec, [gazeX_label, gazeY_label, config.num_gaze_vectors], tf.float32)
      decoded_gaze_input.set_shape(shape)
      #decoded_gaze_input = tf.squeeze(decoded_gaze_input, [0])
      #y_pred = tf.squeeze(y_pred)
      gaze_vec_loss = tf.losses.huber_loss(decoded_gaze_input, y_pred, delta=0.1) # smooth L1 loss

      loss = tf.reduce_mean(gaze_vec_loss)

      init = tf.initialize_all_variables()
      model_summary() # prints model details & parameters
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  # session
  conf = tf.ConfigProto()
  if hasattr(config.eval.execution, 'per_process_gpu_memory_fraction'):
    conf.gpu_options.per_process_gpu_memory_fraction = config.train.execution.per_process_gpu_memory_fraction
  if hasattr(config.eval.execution, 'allow_growth'):
    conf.gpu_options.allow_growth = config.train.execution.allow_growth

  sess = tf.Session(graph=graph, config=conf)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  sess.run(init)

  checkpoints_dir = config.model_dir
  latest_checkpoint = None
  wait_iters = 0

  if not os.path.exists(os.path.join(checkpoints_dir, 'eval')):
    os.mkdir(os.path.join(checkpoints_dir, 'eval'))
  writer = tf.summary.FileWriter(os.path.join(checkpoints_dir, 'eval'), sess.graph)

  old_label_debug = ""
  old_eye_posX_debug = -1.0
  while True:
    if config.eval.checkpoint != '':
      new_checkpoint = config.eval.checkpoint
    else:
      new_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    if latest_checkpoint != new_checkpoint:
      latest_checkpoint = new_checkpoint
      saver.restore(sess, latest_checkpoint)
      current_step = tf.train.load_variable(latest_checkpoint, 'global_step')

      test_size = dataset_size(config.eval.file_list_path)
      print("dataset_size: ")
      print(test_size)
      time_start = time.time()

      cumm_err = 0.0
      cumm_calibrated_err = 0.0
      err_list = []
      calibrated_err_list = []

      steps = int(test_size / config.eval.batch_size) if int(test_size / config.eval.batch_size) else 1
      num = 0

      print("\n\nModel used: ", latest_checkpoint, "\n\n")

      overall_time = 0
      ttt = 0

      # assume subject are sorted in eval.csv
      subject_gaze_labels_list = []
      subject_gaze_inferred_list = []
      folderLabel = b'-1'

      for ii in range(steps):
        st = time.time()
        if(config.eval.debug_show_images):
            curr_val, curr_label, curr_eyeID_label, curr_filename, curr_images, _ = sess.run([y_pred, decoded_gaze_input, eyeID_label, file_name, input_data, file_batch])
        else:
            curr_val, curr_label, curr_eyeID_label, curr_filename, _ = sess.run([y_pred, decoded_gaze_input, eyeID_label, file_name, file_batch])
        if ii > 1000:
          ttt = (time.time() - st)
          overall_time += ttt * 1000
        err, num_ = accuracy(curr_val, curr_label) # raw accuracy - (uncalibrated)
        cumm_err += err
        err_list.append(err)
        num += num_
        if ii % 500 == 0:
          if num != 0:
            print(ii, " out of ", steps, "Accuracy: ", cumm_err / num, '. Runtime: ' + str(overall_time / (ii + 1 - 1000)) + ' ms ' + str(ttt * 1000))

        # DEBUG show, only batchsize 1 supported
        if(err > config.eval.debug_error_threshold):
                if(config.eval.debug_show_images):
                    if(old_eye_posX_debug != curr_label[0][0]):
                        old_eye_posX_debug = curr_label[0][0]
                        print("err_raw: ", err)
                        plt.imshow(curr_images[0].astype(np.float32)[:, :, 0], cmap='gray', interpolation='nearest')
                        plt.show()
        # for now eval supports only pre-sorted lists of subject, like : [sub1, sub1, sub1, sub2, sub2, sub2, ...]
        # it will calculate per-subject accuracies, assuming it is already sorted
        # Also batch sizes other than '1' are not supported
        # if you want to add batch size other than '1' in eval read note below and apply changes:
        ##### note for those who want to chage batch size in eval: ###################
        # curr_filename[0] // 0 is BatchId (first sample in batch).                  #
        # lack of inner for iii loop                                                 #
        # subject_gaze_inferred_list[el][0][1] // second index, middle [0], is Batch #
        # if(curr_eyeID_label[0] == b'R'):                                           #
        ##############################################################################
        if(ii == 0):
            folderLabel = curr_filename[0].split(b'/')[0] # assign first folderLabel as bytes (not string)
            print("Calibrating subject: ", folderLabel.decode("utf-8"))
        if( folderLabel == curr_filename[0].split(b'/')[0]): # as long as it is same subject, add data to the list
            subject_gaze_labels_list.append(curr_label)
            subject_gaze_inferred_list.append(curr_val)
        if ( folderLabel != curr_filename[0].split(b'/')[0] or ii == (steps-1)): #subject has changed or last entry
            # calculate per-subject accuracy after affine transform
            # do not push data yet - label has changed so first calculate, then clear, then push data to empty list
            elems_count = len(subject_gaze_labels_list)
            calib_points_count = min(config.eval.calib_points_count, elems_count)
            step_calib = elems_count // calib_points_count
            inferred_vecs = np.ones([calib_points_count,3], dtype=np.float32) # will be [x y 1]
            label_x_vec = np.empty([calib_points_count], dtype=np.float32)
            label_y_vec = np.empty([calib_points_count], dtype=np.float32)
            el = 0
            for s in range(calib_points_count):
                inferred_vecs[s, 0] = subject_gaze_inferred_list[el][0][0]
                inferred_vecs[s, 1] = subject_gaze_inferred_list[el][0][1]
                label_x_vec[s] = subject_gaze_labels_list[el][0][0]
                label_y_vec[s] = subject_gaze_labels_list[el][0][1]
                el += step_calib
            # calibrate with affine transform
            calib_vec_x, _, _, _ = np.linalg.lstsq(inferred_vecs, label_x_vec)
            calib_vec_y, _, _, _ = np.linalg.lstsq(inferred_vecs, label_y_vec)

            print("calib_vec_x: ", calib_vec_x)
            print("calib_vec_y: ", calib_vec_y)

            err_subject_calib = 0.0
            err_subject__calib_list = []
            num_subject_calib = 0
            err_subject_raw = 0.0
            err_subject__raw_list = []
            num_subject_raw = 0
            for el in range(elems_count):
                curr_calibrated_val = [[subject_gaze_inferred_list[el][0][0]*calib_vec_x[0] +
                                            subject_gaze_inferred_list[el][0][1]*calib_vec_x[1] + calib_vec_x[2],
                                       subject_gaze_inferred_list[el][0][0]*calib_vec_y[0] +
                                                                   subject_gaze_inferred_list[el][0][1]*calib_vec_y[1] + calib_vec_y[2]]]
                tmp_label = [[subject_gaze_labels_list[el][0][0], subject_gaze_labels_list[el][0][1]]]
                err_, num_ = accuracy(curr_calibrated_val, tmp_label) # calibrated acc
                err_raw_, num_raw_ = accuracy([[subject_gaze_inferred_list[el][0][0], subject_gaze_inferred_list[el][0][1]]], tmp_label) # calibrated acc
                err_subject_calib += err_
                calibrated_err_list.append(err_)
                num_subject_calib += num_
                err_subject_raw += err_raw_
                num_subject_raw += num_raw_
                err_subject__calib_list.append(err_)
                err_subject__raw_list.append(err_raw_)

            if num != 0:
              writer.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag="evaluation/acc/calibrated/"+folderLabel.decode("utf-8"), simple_value=float(err_subject_calib / num_subject_calib)),
                                  tf.Summary.Value(tag="evaluation/acc/raw/"+folderLabel.decode("utf-8"), simple_value=float(err_subject_raw / num_subject_raw))]),
                                  current_step)
              print("SubjectID: ", folderLabel.decode("utf-8") ," Acc Calibrated: ", err_subject_calib / num_subject_calib)
              print("SubjectID: ", folderLabel.decode("utf-8") ," Acc (Raw): ", err_subject_raw / num_subject_raw)
              log_histogram(writer, "evaluation/histogram/calibrated/"+folderLabel.decode("utf-8"), err_subject__calib_list, current_step)
              log_histogram(writer, "evaluation/histogram/raw/"+folderLabel.decode("utf-8"), err_subject__raw_list, current_step)
              cumm_calibrated_err += err_subject_calib

            # change folderLabel and clear subject's data
            folderLabel = curr_filename[0].split(b'/')[0]
            subject_gaze_labels_list.clear()
            subject_gaze_inferred_list.clear()
            # also push new data
            subject_gaze_labels_list.append(curr_label) # no need in ii == (steps-1) case but also not harmful
            subject_gaze_inferred_list.append(curr_val) # no need in ii == (steps-1) case but also not harmful

      writer.add_summary(
        tf.Summary(value=[tf.Summary.Value(tag='evaluation/acc_raw', simple_value=float(cumm_err / num)),
                          tf.Summary.Value(tag='evaluation/acc_calibrated', simple_value=float(cumm_calibrated_err / num))]),
        current_step)
      log_histogram(writer, "evaluation/histogram/calibrated", calibrated_err_list, current_step)
      log_histogram(writer, "evaluation/histogram/raw", err_list, current_step)
      print('Test acc raw: {}'.format(cumm_err / num))
      print('Test acc calibrated: {}'.format(cumm_calibrated_err / num))
      print('Time per step: {} for test size {}'.format(time.time() - time_start / steps, test_size))

    else:
      if wait_iters % 12 == 0:
        sys.stdout.write('\r')
        for _ in range(11 + wait_iters // 12):
          sys.stdout.write(' ')
        sys.stdout.write('\r')
        for _ in range(1 + wait_iters // 12):
          sys.stdout.write('|')
      else:
        sys.stdout.write('.')
      sys.stdout.flush()
      time.sleep(5)
      wait_iters += 1
    if config.eval.checkpoint != '':
      break


  coord.request_stop()
  coord.join(threads)
  sess.close()

def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  validate(cfg)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
