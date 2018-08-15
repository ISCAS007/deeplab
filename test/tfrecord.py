# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Converts Cityscapes data to TFRecord file format with Example protos.

The Cityscapes dataset is expected to have the following directory structure:
  - build_cityscapes_data.py (current working directiory).
  - build_data.py
  + cityscapes
     + cityscapesscripts
       + annotation
       + evaluation
       + helpers
       + preparation
       + viewer
     + gtFine
       + train
       + val
       + test
     + leftImg8bit
       + train
       + val
       + test
     + tfrecord

This script converts data into sharded data files and save at tfrecord folder.

Note that before running this script, the users should (1) register the
Cityscapes dataset website at https://www.cityscapes-dataset.com to
download the dataset, and (2) run the script provided by Cityscapes
`preparation/createTrainIdLabelImgs.py` to generate the training groundtruth.

Also note that the tensorflow model will be trained with `TrainId' instead
of `EvalId' used on the evaluation server. Thus, the users need to convert
the predicted labels to `EvalId` for evaluation on the server. See the
vis.py for more details.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import glob
import math
import os.path
import re
import sys
import cv2
from deeplab.datasets import build_data
from deeplab.datasets import segmentation_dataset
from deeplab import common
from src.utils.disc import get_edge
import tensorflow as tf
slim = tf.contrib.slim
dataset_data_provider = slim.dataset_data_provider

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('cityscapes_root',
                           'deeplab/datasets/cityscapes',
                           'Cityscapes dataset root folder.')

tf.app.flags.DEFINE_string(
    'output_dir',
    '/tmp/tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')


_NUM_SHARDS = 2

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}

# Image file pattern.
_IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP['image'])


def _get_files(data, dataset_split):
  """Gets files for the specified data type and dataset split.

  Args:
    data: String, desired data ('image' or 'label').
    dataset_split: String, dataset split ('train', 'val', 'test')

  Returns:
    A list of sorted file names or None when getting label for
      test set.
  """
  if data == 'label' and dataset_split == 'test':
    return None
  pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
  search_files = os.path.join(
      FLAGS.cityscapes_root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
  filenames = glob.glob(search_files)
  return sorted(filenames)


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, val).

  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  image_files = _get_files('image', dataset_split)
  label_files = _get_files('label', dataset_split)
  
  image_files=image_files[:10]
  label_files=image_files[:10]
  
  num_images = len(image_files)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('png', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        dataset_split, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(FLAGS.output_dir, shard_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_data = tf.gfile.FastGFile(image_files[i], 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_data = tf.gfile.FastGFile(label_files[i], 'rb').read()
#        print('seg_data type',type(seg_data))
        label=cv2.imread(label_files[i],cv2.IMREAD_GRAYSCALE)
        edge=get_edge(label)
#        edge_bytes=tf.train.BytesList(value=edge)
#        edge_feature=tf.train.Feature(int64_list=tf.train.Int64List(value=edge.reshape(-1)))
        edge_feature=tf.train.Feature(float_list=tf.train.FloatList(value=edge))
        
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        re_match = _IMAGE_FILENAME_RE.search(image_files[i])
        if re_match is None:
          raise RuntimeError('Invalid image filename: ' + image_files[i])
        filename = os.path.basename(re_match.group(1))
        example = build_data.image_seg_edge_to_tfexample(
            image_data, filename, height, width, seg_data, edge_feature)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def _load_dataset(dataset_split):
    dataset = segmentation_dataset.get_dataset(
          'cityscapes', dataset_split, dataset_dir=FLAGS.output_dir)
    
    is_training=True
    num_readers=1
    data_provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      num_epochs=None if is_training else 1,
      shuffle=is_training)
    
#    items=['image','image_name','height','width','labels_class']
#    for item in items:
#        print(data_provider.get([item]))
        
    image, height, width, label ,edge = data_provider.get(
      [common.IMAGE, common.HEIGHT, common.WIDTH, common.LABELS_CLASS, common.EDGE])
    
    print('image',type(image),image.shape)
    print('label',type(label),label.shape)
    print('edge',type(edge),edge.shape)
    
    sess=tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess)
    
    np_image=sess.run(image)
    np_label=sess.run(label)
    np_edge=sess.run(edge)
    print('np_image',type(np_image),np_image.shape)
    print('np_label',type(np_label),np_label.shape)
    print('np_edge',type(np_edge),np_edge.shape)
    
    
def main(unused_argv):
  # Only support converting 'train' and 'val' sets for now.
  for dataset_split in ['train', 'val']:
    _convert_dataset(dataset_split)

  for dataset_split in ['train','val']:
      _load_dataset(dataset_split)

if __name__ == '__main__':
  tf.app.run()
