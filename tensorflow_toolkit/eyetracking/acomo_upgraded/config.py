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

input_shape = (127, 127, 1)  # (height, width, channels)

num_gaze_vectors = 2                    # num of outputs, 2 floats: gazeX, gazeY
use_coord_conv = True                   # https://arxiv.org/abs/1807.03247
use_global_context = True               # globalContext as described in https://arxiv.org/pdf/1904.11492.pdf
use_self_attention = True               # for the last layer sth similar to https://arxiv.org/abs/1706.03762
use_fireblocks = True                   # squeezenet-like technique, taken from https://github.com/opencv/openvino_training_extensions/tree/develop/tensorflow_toolkit/lpr
first_layer_channels = 16               # each next channel will have 1.5x feature maps


accuracy_measure = "acomo"              # "acomo" : gaze angle as ordered extrinsic yaw, pitch rotations
                                        # "nvgaze" : gaze angle as projection to perpendicular planes

# Path to the folder where all training and evaluation artifacts will be located
model_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_acomo_upgraded'))
if not os.path.exists(model_dir):
  os.makedirs(model_dir)


class train:
  # Path to annotation file with training data in per line format: <path_to_image_with_license_plate label>
  file_list_path = '/media/kamil/DATA/ACOMO-DATASETS/Train6-merged-monocular-R-shuffled-train.csv'

  batch_size = 64
  steps = 1000000
  learning_rate = 0.0001
  grad_noise_scale = 0.001
  opt_type = 'Adam'

  save_checkpoints_steps = 1000      # Number of training steps when checkpoint should be saved
  display_iter = 10

  apply_basic_aug = True
  apply_stn_aug = False
  apply_coarse_dropout = True
  apply_blur_aug = False

  need_to_save_weights = True
  need_to_save_log = True

  debug_show_images = False           # Turn on to see how images looks like after augmentation

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations


class eval:
  # Path to annotation file with validation data in per line format: <path_to_image_with_license_plate label>
  calib_points_count = 10000
  file_list_path = '/media/kamil/DATA/ACOMO-DATASETS/Train6-merged-monocular-R-shuffled-test-SMALL.csv'
  checkpoint = ''
  batch_size = 1    # other batch size not supported, see eval.py

  debug_show_images = False           # Turn on to see how images looks like after augmentation
  debug_error_threshold = 30.0         # error threshold of images to display

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations


class infer:
  # Path to text file with list of images in per line format: <path_to_image_with_license_plate>
  file_list_path = '/media/kamil/DATA/ACOMO-DATASETS/BraciszkiTrain-merged-monocular-R-shuffled-test.csv'
  checkpoint = ''
  batch_size = 1

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations
