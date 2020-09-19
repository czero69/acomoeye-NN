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

import tensorflow as tf
import tensorflow.contrib.slim as slim
from .coordConv import CoordConv
import math

# get Channel Count for a layer
def getC(first_layer_channels, layer_id = 0):
    new_chann_count = first_layer_channels
    for i in range(layer_id):
        new_chann_count *= 1.5
    return math.ceil(new_chann_count)

class EyeNet:
  def __init__(self, first_layer_channels, do_coordConv, do_globalContext, do_fireBlocks, do_selfAttention):
      super(EyeNet, self).__init__()
      self.fl_c = first_layer_channels
      self.coord_conv = do_coordConv
      self.global_context = do_globalContext
      self.fire_blocks = do_fireBlocks
      self.self_attention = do_selfAttention
      # basic_block = fire_block
      if(self.fire_blocks):
          self.basic_block = self.small_fire_block_valid
      else:
          self.basic_block = self.just_conv_block_valid

  # Fire block
  @staticmethod
  def fire_block(block_input, outputs):
    fire = slim.conv2d(block_input, outputs / 4, [1, 1])
    fire = slim.conv2d(fire, outputs / 4, [3, 3])
    fire = slim.conv2d(fire, outputs, [1, 1])
    return fire

  # Small Fire block
  @staticmethod
  def small_fire_block(block_input, outputs):
    fire = slim.conv2d(block_input, outputs / 4, [1, 1])
    fire = slim.conv2d(fire, outputs / 4, [3, 1])
    fire = slim.conv2d(fire, outputs / 4, [1, 3])
    fire = slim.conv2d(fire, outputs, [1, 1])
    return fire

  # Small Fire block VALID
  @staticmethod
  def small_fire_block_valid(self, block_input, outputs):
    fire = slim.conv2d(block_input, math.ceil(outputs/4), [1, 1])
    fire = slim.conv2d(fire, math.ceil(outputs/4), [3, 1], [2, 1], padding='VALID')
    fire = slim.conv2d(fire, math.ceil(outputs/4), [1, 3], [1, 2], padding='VALID')
    fire = slim.conv2d(fire, outputs, [1, 1])
    return fire

  # Small Fire block VALID
  @staticmethod
  def small_fire_block_valid_V2(self, block_input, outputs):
    fire = slim.conv2d(block_input, math.ceil(outputs/4), [1, 1])
    fire = slim.conv2d(fire, math.ceil(outputs/4), [3, 3], [2, 2], padding='VALID')
    fire = slim.conv2d(fire, outputs, [1, 1])
    return fire

  # Small Fire block VALID
  @staticmethod
  def just_conv_block_valid(self, block_input, outputs):
    cnn = slim.conv2d(block_input, outputs, [3, 3], [2, 2], padding='VALID')
    return cnn

  # Inception-ResNet like block
  @staticmethod
  def resinc_block(block_input, outputs):
    inputs = int(block_input.get_shape()[3])
    if inputs == outputs:
      res = block_input
    else:
      res = slim.conv2d(block_input, outputs, [1, 1])
    inc1 = slim.conv2d(block_input, outputs / 8, [1, 1])
    inc2 = slim.conv2d(block_input, outputs / 8, [1, 1])
    inc2 = slim.conv2d(inc2, outputs / 8, [3, 1])
    inc2 = slim.conv2d(inc2, outputs / 8, [1, 3])
    conc = tf.concat(3, [inc1, inc2])
    inc = slim.conv2d(conc, outputs, [1, 1])
    return res + inc

  # basic_block = resinc_block

  # Convolution block for CNN
  @staticmethod
  def convolution_block(self, block_input, outputs, stride, **kwargs):
    scope = kwargs.pop('scope', None)
    # cr = slim.conv2d(input, outputs, [3, 3], scope=scope)
    b_block = self.basic_block(self, block_input, outputs)
    max_pool = slim.max_pool2d(b_block, [3, 3], stride=(stride, 1), padding='VALID', scope=scope)
    return max_pool

  @staticmethod
  def enet_input_block(self, block_input, **kwargs):
    scope = kwargs.pop('scope', None)
    input1 = slim.conv2d(block_input, 61, [3, 3], stride=(2, 1), padding='VALID', scope=scope)
    input2 = slim.avg_pool2d(block_input, [3, 3], stride=(2, 1), padding='VALID', scope=scope)
    step1 = tf.concat(3, [input1, input2])
    step2 = self.basic_block(self, step1, 128)
    step2 = slim.max_pool2d(step2, [2, 2], stride=(1, 1), padding='VALID', scope=scope)
    return step2

  @staticmethod
  def std_input_block(self, block_input):
    return slim.stack(block_input, EyeNet.convolution_block, [(64, 1), (128, 2)])

  @staticmethod
  def mixed_input_block(self, block_input):
    cnn = slim.conv2d(block_input, 64, [3, 3])
    cnn = slim.max_pool2d(cnn, [3, 3], stride=(1, 1), padding='VALID')
    cnn = self.basic_block(self, cnn, 128)
    cnn = slim.max_pool2d(cnn, [3, 3], stride=(2, 1), padding='VALID')
    return cnn

  @staticmethod
  def eye_input_block(self, block_input):
    cnn = slim.conv2d(block_input,  self.fl_c, [3, 3], [2, 2], padding='VALID')
    cnn = self.basic_block(self, cnn, getC(self.fl_c,1))  # [3,3] , padding='VALID'
    return cnn

  @staticmethod
  def eye_coord_input_block(self, block_input):

    coord_layer = CoordConv(1, 1, False, self.fl_c, 3, 2, activation=tf.nn.relu)
    cnn = coord_layer(block_input)
    cnn = self.basic_block(self, cnn, getC(self.fl_c,1))  # [3,3] , padding='VALID'
    return cnn

  @staticmethod
  def GC_block_old(self, block_input):

    selected_feature_maps = block_input[:,:,:,18:24] # 6 last feature maps NHWC
    pattern = slim.fully_connected(slim.flatten(selected_feature_maps), self.fl_c)   # 1d global context vector descriptor
    pattern = tf.reshape(pattern, (-1, 1, 1, self.fl_c))
    width = int(block_input.get_shape()[2])
    heihgt = int(block_input.get_shape()[1])
    pattern = tf.tile(pattern, [1, heihgt, width, 1])
    cnn = tf.concat(axis=3, values=[block_input, pattern])
    return cnn

  # from the paper https://arxiv.org/pdf/1904.11492.pdf
  @staticmethod
  def GC_block(self, block_input, reduction_ratio = 16):

    N = block_input.shape[0].value
    C = block_input.shape[3].value
    norm_layer = tf.keras.layers.LayerNormalization(axis=(1,2,3))
    my_relu = tf.keras.layers.ReLU()
    attention = slim.conv2d(block_input, 1, [1, 1], activation_fn=tf.nn.softmax)
    block_input_flattened = tf.reshape(block_input, [N,-1,C])
    attention = tf.squeeze(attention, axis=[3])
    attention_flattened = tf.reshape(attention, [N,-1])

    c11 = tf.einsum('bfc,bf->bc', block_input_flattened, attention_flattened)
    c11 = tf.reshape(c11, (N, 1, 1, C))
    c12 = slim.conv2d(c11, math.ceil(C/reduction_ratio), [1, 1], activation_fn=None)
    c15 = slim.conv2d(my_relu(norm_layer(c12)), C, [1, 1], activation_fn=None)
    cnn = tf.math.add(block_input, c15)
    return cnn

  @staticmethod
  def self_attention(self, block_input, reduction_ratio = 16):
      N = block_input.shape[0].value
      C = block_input.shape[3].value
      norm_layer = tf.keras.layers.LayerNormalization(axis=(1,2,3))
      my_relu = tf.keras.layers.ReLU()
      decoded = slim.conv2d(block_input, math.ceil(C/reduction_ratio), [1, 1], activation_fn=None)
      encoded = slim.conv2d(my_relu(norm_layer(decoded)), C, [1, 1], activation_fn=tf.nn.softmax)
      cnn = tf.math.multiply(block_input, encoded)
      return cnn

  input_block = eye_coord_input_block

  @staticmethod
  def eyenet(self, net_input):
    with slim.arg_scope([slim.fully_connected, slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):

      if(self.coord_conv):
          cnn = EyeNet.eye_coord_input_block(self, net_input)
      else:
          cnn = EyeNet.eye_input_block(self, net_input)

      if(self.global_context):
          cnn = EyeNet.GC_block(self, cnn)
      cnn = self.basic_block(self, cnn, getC(self.fl_c,2))
      cnn = self.basic_block(self, cnn, getC(self.fl_c,3))
      if(self.global_context):
          cnn = EyeNet.GC_block(self, cnn)
      cnn = self.basic_block(self, cnn, getC(self.fl_c,4))
      cnn = slim.dropout(cnn)
      cnn = self.basic_block(self, cnn, getC(self.fl_c,5))
      cnn = slim.dropout(cnn)
      if(self.self_attention):
          cnn = EyeNet.self_attention(self,cnn)

      return cnn
