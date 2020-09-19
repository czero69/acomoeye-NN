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

import os
import re
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from et.networks.eyenet import EyeNet
from spatial_transformer import transformer
import imgaug.augmenters as iaa


class InputData:
  # pylint: disable=too-many-arguments
  def __init__(self, batch_size, input_shape, file_list_path,
               apply_basic_aug=False, apply_stn_aug=True, apply_coarse_dropout=False, apply_blur_aug=False):
    self.batch_size = batch_size
    self.input_shape = input_shape
    self.file_list_path = file_list_path
    self.apply_basic_aug = apply_basic_aug
    self.apply_stn_aug = apply_stn_aug
    self.apply_coarse_dropout = apply_coarse_dropout
    self.apply_blur_aug = apply_blur_aug

  def input_fn(self):
    file_src = tf.train.string_input_producer([self.file_list_path])
    base_path = os.path.dirname(self.file_list_path)
    # image, label, country_code = read_data(self.batch_size, self.input_shape, file_src)
    image, eyeID, gazeX, gazeY = read_data(self.batch_size, self.input_shape, file_src, base_path)

    if self.apply_basic_aug:
      image = augment(image)

    if self.apply_coarse_dropout:
      image = augment_coarse_droput(image)

    if self.apply_stn_aug:
      image = tf.cond(tf.constant(True, dtype=tf.bool), lambda: image, lambda: augment_with_stn(image))

    # blur/sharpen augmentation
    if self.apply_blur_aug:
      data, = tf.py_func(random_blur, [image], [tf.float32])
      data.set_shape([self.batch_size] + list(self.input_shape))  # [batch_size, height, width, channels_num]
    else:
      data = image

    return data, eyeID, gazeX, gazeY


def read_data(batch_size, input_shape, file_src, base_path):
  reader = tf.TextLineReader()
  _, value = reader.read(file_src)
  # filename, label, country_code = tf.decode_csv(value, [[''], [''], ['']], ' ')
  filename, eyeID, gazeX, gazeY = tf.decode_csv(value, [[''], [''], [''], ['']], ',')
  image_file = tf.read_file(base_path + '/' + filename)

  height, width, channels_num = input_shape
  rgb_image = tf.image.decode_png(image_file, channels=channels_num)
  #rgb_image = tf.image.decode_jpeg(image_file, channels=channels_num)
  rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
  resized_images = tf.image.resize_images(rgb_image_float, [height, width])
  resized_images.set_shape(input_shape)

  min_after_dequeue = 4000*4 # 4000: 640x480 jpegs limit for 32GB;
  capacity = min_after_dequeue + 3 * batch_size
  image_batch, eyeID_label_batch, gazeX_batch, gazeY_batch = tf.train.shuffle_batch([resized_images, eyeID, gazeX, gazeY], batch_size=batch_size, capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)
  return image_batch, eyeID_label_batch, gazeX_batch, gazeY_batch


def iaa_augment(images):
    seq = iaa.Sequential([
        iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.0))), # blur images with a sigma of 0 to 3.0
        iaa.Sometimes(0.5, iaa.arithmetic.CoarseDropout(0.02, size_percent=0.1))
    ])
    images_aug = seq(images=images)
    return images_aug
def augment_coarse_droput(images):
    shape = images.get_shape()
    images = tf.numpy_function(iaa_augment, [images], tf.float32)
    images.set_shape(shape)
    return images

# Function for basic image augmentation - photometric distortions
def augment(images):
  print("STANDARD AUGMENT ON")
  augmented = tf.image.random_brightness(images, max_delta=0.2)
  augmented = tf.image.random_contrast(augmented, lower=0.8, upper=1.2)
  augmented = tf.add(augmented, tf.truncated_normal(tf.shape(augmented), stddev=0.005))
  return images


# Function for STN image augmentation - geometric transformations with STN
def augment_with_stn(images):
  print("augment_with_stn AUGMENT ON")
  #identity = identity_transform(images)
  #noise = tf.truncated_normal(identity.get_shape(), stddev=0.1)
  #return apply_stn(images, tf.add(identity, noise))

  # curriculum_rate = tf.clip_by_value(0.0001 * tf.cast(global_step, tf.float32), 0.0, 1.0)
  # noise = tf.scalar_mul(curriculum_rate, noise)
  return apply_stn(images, locnet_block(images))

def locnet_block(images):
  input1 = slim.avg_pool2d(images, [3, 3], stride=2)

  input1 = slim.conv2d(input1, 32, [5, 5], stride=3)
  input1 = slim.conv2d(input1, 32, [1, 1], stride=4)

  input2 = slim.conv2d(images, 32, [5, 5], stride=5)
  input2 = slim.conv2d(input2, 32, [1, 1], stride=5)

  step1 = tf.concat([input1, input2], axis=-1)
  #step1 = input1

  step2 = slim.dropout(step1)

  step2 = tf.contrib.layers.flatten(step2)

  step3 = slim.fully_connected(step2, num_outputs=32, activation_fn=tf.nn.tanh)
  step3 = slim.fully_connected(step3, num_outputs=6, activation_fn=tf.nn.tanh)

  return step3

# Function for identity transformation
def identity_transform(images):
  shape = images.get_shape()
  ident = tf.constant(np.array([[[1., 0, 0], [0, 1., 0]]]).astype('float32'))
  return tf.tile(ident, [shape[0].value, 1, 1])

# Function wrapper for STN application
def apply_stn(images, transform_params):
  shape = images.get_shape()
  out_size = (shape[1], shape[2])
  warped = transformer(images, transform_params, out_size)
  warped.set_shape(shape)
  return warped


def random_blur(images):
  result = []
  for k in range(images.shape[0]):
    samples = np.random.normal(scale=1.)
    kernel = np.array([[0., samples, 0.], [samples, 1. - 4. * samples, samples], [0., samples, 0.]])
    result.append(cv2.filter2D(images[k], -1, kernel).astype(np.float32))
  return np.array(result)


# Function for construction whole network
class InferenceEngine:
    def __init__(self, first_layer_channels, do_coordConv, do_globalContext, do_fireBlocks, do_selfAttention):
        super(InferenceEngine, self).__init__()
        self.first_layer_channels = first_layer_channels
        self.coord_conv = do_coordConv
        self.global_context = do_globalContext
        self.fire_blocks = do_fireBlocks
        self.EyeNet = EyeNet(first_layer_channels=first_layer_channels, do_coordConv=do_coordConv,
                                   do_globalContext=do_globalContext, do_fireBlocks = do_fireBlocks, do_selfAttention = do_selfAttention)

    def __call__(self, input, num_gaze_vectors):
      cnn = self.EyeNet.eyenet(self.EyeNet, net_input = input)

      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          normalizer_fn=slim.batch_norm,
                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                          weights_regularizer=slim.l2_regularizer(0.0005)):
        # logits = slim.fully_connected(slim.flatten(classes), 2)
        #logits = slim.fully_connected(tf.contrib.layers.flatten(cnn), num_outputs=num_gaze_vectors, activation_fn=None) # None == linear activation
        logits = tf.layers.dense(tf.contrib.layers.flatten(cnn), num_gaze_vectors, activation='linear')
      return logits

class GazeVecUtils:
  country_codes = {}
  r_country_codes = {}

  # Generate CTC from country_codes
  @staticmethod
  def decode_gaze_vec(gazeX_list, gazeY_list, num_gaze_outputs):
    batch_size = gazeX_list.shape[0]
    x_val = []

    for batch_i in range(batch_size):
      temp = np.empty(shape=num_gaze_outputs, dtype=np.float32)  # 3 element array
      temp[0] = np.float32(float(gazeX_list[batch_i]))
      temp[1] = np.float32(float(gazeY_list[batch_i]))
      x_val.append(temp)

    return np.array(x_val, dtype=np.float32)
