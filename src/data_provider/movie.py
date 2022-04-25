# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MOVIE"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class InputHandle(object):
  """Class for handling dataset inputs."""

  def __init__(self, datas, indices, input_param):
    self.name = input_param['name']
    self.input_data_type = input_param.get('input_data_type', 'float32')
    self.minibatch_size = input_param['minibatch_size']
    self.image_width = input_param['image_width']
    self.datas = datas
    self.indices = indices
    self.current_position = 0
    self.current_batch_indices = []
    self.current_input_length = input_param['seq_length']

  def total(self):
    return len(self.indices)

  def begin(self, do_shuffle=True):
    # logger.info('Initialization for read data ')
    if do_shuffle:
      random.shuffle(self.indices)
    self.current_position = 0
    self.current_batch_indices = self.indices[self.current_position:self.current_position +self.minibatch_size]

  def next(self):
    self.current_position += self.minibatch_size
    if self.no_batch_left():
      return None
    self.current_batch_indices = self.indices[self.current_position:self.current_position +self.minibatch_size]

  def no_batch_left(self):
    if self.current_position + self.minibatch_size >= self.total():
      return True
    else:
      return False

  def get_batch(self):
    """Gets a mini-batch."""
    if self.no_batch_left():
      logger.error(
          'There is no batch left in %s.'
          'Use iterators.begin() to rescan from the beginning.',
          self.name)
      return None
    input_batch = np.zeros(
        (self.minibatch_size, self.current_input_length, self.image_width,
         self.image_width, 3)).astype(self.input_data_type)
    for i in range(self.minibatch_size):
      batch_ind = self.current_batch_indices[i]
      begin = batch_ind
      end = begin + self.current_input_length
      data_slice = self.datas[begin:end, :, :, :]
      input_batch[i, :self.current_input_length, :, :, :] = data_slice
      # logger.info('data_slice shape')
      # logger.info(data_slice.shape)
      # logger.info(input_batch.shape)
    input_batch = input_batch.astype(self.input_data_type)
    return input_batch

  def print_stat(self):
    print('Iterator Name: %s', self.name)
    print('    current_position: %s', str(self.current_position))
    print('    Minibatch Size %s: ', str(self.minibatch_size))
    print('    total Size: %s', str(self.total()))
    print('    current_input_length: %s', str(self.current_input_length))
    print('    Input Data Type: %s', str(self.input_data_type))


class DataProcess(object):
  """Class for preprocessing dataset inputs."""

  def __init__(self, input_param):
    self.paths = input_param['paths']
    self.category = ['CA_1750', 'CA_0020', 'CA_1720','CA_1750'] #'CA_0780', 'CA_0790', 'CA_0800', 'CA_1120', ]
    self.image_width = input_param['image_width']

    # 3->2 as a sequence of 5
    # roughly traning vs testing = 8:2
    self.train_frames = {
        'CA_0020': [80, 100],  # 102 frames. 20 sequences.  
        'CA_0780': [160, 200], # 203 frames. 
        'CA_0790': [275, 345], # 347 frames
        'CA_0800': [165, 205], # 208 frames
        'CA_1120': [180, 225], # 229 frames
        'CA_1720': [70, 90],  # 90 frames
        'CA_1750': [40, 50],  # 54 framess
    }

    self.input_param = input_param
    self.seq_len = input_param['seq_length']

  def load_data(self, paths, mode='train'):
    """Loads the dataset.

    Args:
      paths: List of action_path.
      mode: Training or testing.

    Returns:
      A dataset and indices of the sequence.
    """
    path = paths[0]
    if mode == 'train':
      isTrain = True 
    elif mode == 'test':
      isTrain = False 
    else:
      print('ERROR!')

    frames_np = []
    frames_file_name = []
    frames_person_mark = []
    frames_category = []
    person_mark = 0

    c_dir_list = self.category
    for c_dir in c_dir_list:  
        c_dir_path = os.path.join(path, c_dir)
        p_c_dir_list = os.listdir(c_dir_path)
        person_mark += 1

        for img_dir in p_c_dir_list:  
            curr_img_n = int(img_dir[:3])
            if isTrain:
              if curr_img_n > self.train_frames[c_dir][0]:
                continue
            elif curr_img_n < self.train_frames[c_dir][0] or curr_img_n >= self.train_frames[c_dir][1]:
              continue
            img_path = os.path.join(c_dir_path, img_dir)
            
            frame_np = cv2.imread(img_path) / 255
            # frame_np = cv2.cvtColor(frame_im, cv2.COLOR_BGR2RGB)
            # cv2.imshow('image', frame_im)
            # cv2.waitKey(0)
            # frame_np = np.array(frame_im)  # (1000, 1000) numpy array
            # print(frame_np.shape)
            # frame_np = frame_np[:, :, 0]  #
            frames_np.append(frame_np)
            frames_file_name.append(img_dir)
            frames_person_mark.append(person_mark)
    
    # is it a begin index of sequence
    indices = []
    index = len(frames_person_mark) - 1
    while index >= self.seq_len - 1:
        if frames_person_mark[index] == frames_person_mark[index - self.seq_len +1]:
            end = int(frames_file_name[index][:3])
            start = int(frames_file_name[index - self.seq_len + 1][:3])
            if end - start == self.seq_len - 1:
                indices.append(index - self.seq_len + 1)
        index -= 1

    frames_np = np.asarray(frames_np)
    data = np.zeros((frames_np.shape[0], self.image_width, self.image_width, 3))
    for i in range(len(frames_np)):
      temp = np.float32(frames_np[i, :, :])
      data[i, :, :, :] = temp
    print('there are ' + str(data.shape[0]) + ' pictures')
    print('there are ' + str(len(indices)) + ' sequences')
    return data, indices

  def get_train_input_handle(self):
    train_data, train_indices = self.load_data(self.paths, mode='train')
    print(f'Traning data: {train_data.shape}')
    return InputHandle(train_data, train_indices, self.input_param)

  def get_test_input_handle(self):
    test_data, test_indices = self.load_data(self.paths, mode='test')
    return InputHandle(test_data, test_indices, self.input_param)
