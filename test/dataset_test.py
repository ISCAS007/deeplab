# -*- coding: utf-8 -*-
"""
cityscapes: /home/yzbx/git/gnu/models/research/deeplab/datasets/cityscapes/tfrecord
pascal_voc_seg: /home/yzbx/git/gnu/models/research/deeplab/datasets/pascal_voc_seg/tfrecord
"""
from datasets import segmentation_dataset
import tensorflow as tf

slim = tf.contrib.slim
dataset_data_provider = slim.dataset_data_provider
#import argparse
flags = tf.app.flags

FLAGS = flags.FLAGS

# Dataset settings.
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', 'datasets/pascal_voc_seg/tfrecord', 'Where the dataset reside.')


def main(unused_argv):
    dataset = segmentation_dataset.get_dataset(
          FLAGS.dataset, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)
    
    is_training=True
    num_readers=1
    data_provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      num_epochs=None if is_training else 1,
      shuffle=is_training)
    
    items=['image','image_name','height','width','labels_class']
    for item in data_provider.list_items():
        print(item)
#        print(data_provider.get([item]))
#    samples = input_generator.get(
#          dataset,
#          FLAGS.train_crop_size,
#          clone_batch_size,
#          min_resize_value=FLAGS.min_resize_value,
#          max_resize_value=FLAGS.max_resize_value,
#          resize_factor=FLAGS.resize_factor,
#          min_scale_factor=FLAGS.min_scale_factor,
#          max_scale_factor=FLAGS.max_scale_factor,
#          scale_factor_step_size=FLAGS.scale_factor_step_size,
#          dataset_split=FLAGS.train_split,
#          is_training=True,
#          model_variant=FLAGS.model_variant)
    
if __name__ == '__main__':
#    flags.mark_flag_as_required('dataset_dir')
    tf.app.run()