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
import re
#from lpr.trainer import encode, decode_beams, decode_country_code
from et.trainer import GazeVecUtils
import numpy as np
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def yaw_matrix(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                     [math.sin(yaw), math.cos(yaw), 0.0],
                     [0.0, 0.0, 1.0]])

def pitch_matrix(pitch):
    return np.array([[math.cos(pitch), 0.0, math.sin(pitch)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(pitch), 0.0, math.cos(pitch)]])

def roll_matrix(roll):
    return np.array([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])

def dataset_size(fname):
  count = 0
  with open(fname, 'r') as file_:
    for _ in file_:
      count += 1
  return count

def edit_distance(string1, string2):
  len1 = len(string1) + 1
  len2 = len(string2) + 1
  tbl = {}
  for i in range(len1):
    tbl[i, 0] = i
  for j in range(len2):
    tbl[0, j] = j
  for i in range(1, len1):
    for j in range(1, len2):
      cost = 0 if string1[i - 1] == string2[j - 1] else 1
      tbl[i, j] = min(tbl[i, j - 1] + 1, tbl[i - 1, j] + 1, tbl[i - 1, j - 1] + cost)

  return tbl[i, j]

# coordinates like that:
#  z
#  |  y
#  | /
#  |/--->x

# for measuring nvidia nvgaze, gaze angle as projection to perpendicular planes
def accuracy_projection(curr_val, curr_label):
  label_len = len(curr_label)

  err = 0
  num = 0

  for i in range(label_len): # iterate batches

      # gaze angles are horizontal and vertical angles as 'one-sided visual angle'
      gaze_val = [math.tan(curr_val[i][0]), math.tan(curr_val[i][1]), 1]
      gaze_label = [math.tan(curr_label[i][0]), math.tan(curr_label[i][1]), 1]

      # angle between val and label (angular error in radians)
      e = math.atan2(np.linalg.norm(np.cross(gaze_label,gaze_val)),np.dot(gaze_label,gaze_val));
      e_degrees = math.degrees(e)
      err += e_degrees
      num += 1
  return float(err), num


# coordinates like that:
#  z
#  |  y
#  | /
#  |/--->x

# for measuring acomo gaze angle as ordered yaw, pitch rotation
def accuracy_yaw_pitch(curr_val, curr_label):
    label_len = len(curr_label)

    err = 0
    num = 0
    AXIS_Z = [0,0,1] # for x-horizontal-rotations
    AXIS_X = [1,0,0] # for y-vertical-rotations
    AXIS_Y = [0,1,0] # for y-vertical-rotations
    gazeX_init_val = [1,0,0]
    gazeY_init_val = [0,1,0]
    #print(curr_label)
    #print(curr_val)
    for i in range(label_len): # iterate batches

        ### -- this is alternative way to make rotations : --
        #val_x_around_z = np.dot(rotation_matrix(AXIS_Z, curr_val[i][0]), gazeX_init_val)
        #val_y_around_z = np.dot(rotation_matrix(AXIS_Z, curr_val[i][0]), gazeY_init_val)
        #gaze_val = np.dot(rotation_matrix(val_y_around_z, -1*curr_val[i][1]), val_x_around_z)

        #lab_x_around_z = np.dot(rotation_matrix(AXIS_Z, curr_label[i][0]), gazeX_init_val)
        #lab_y_around_z = np.dot(rotation_matrix(AXIS_Z, curr_label[i][0]), gazeY_init_val)
        #gaze_label = np.dot(rotation_matrix(lab_y_around_z, -1*curr_label[i][1]), lab_x_around_z)
        ### -- alternative way to make rotations --

        ###
        R_val = np.dot(yaw_matrix(curr_val[i][0]) , pitch_matrix(curr_val[i][1]))
        R_lab = np.dot(yaw_matrix(curr_label[i][0]) , pitch_matrix(curr_label[i][1]))

        gaze_val = np.dot(R_val, gazeX_init_val)
        gaze_label = np.dot(R_lab, gazeX_init_val)
        ###
        # angle between val and label (angular error in radians)
        e = math.atan2(np.linalg.norm(np.cross(gaze_label,gaze_val)),np.dot(gaze_label,gaze_val));
        e_degrees = math.degrees(e)
        err += e_degrees
        num += 1
    return float(err), num
