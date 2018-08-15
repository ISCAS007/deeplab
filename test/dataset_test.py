# -*- coding: utf-8 -*-
"""
cityscapes: /home/yzbx/git/gnu/models/research/deeplab/deeplab/datasets/cityscapes/tfrecord
pascal_voc_seg: /home/yzbx/git/gnu/models/research/deeplab/deeplab/datasets/pascal_voc_seg/tfrecord
"""
from deeplab.datasets import segmentation_dataset
import tensorflow as tf
import numpy as np
from deeplab import input_preprocess
from deeplab import common
from src.utils.disc import batch_get_edge

slim = tf.contrib.slim
prefetch_queue = slim.prefetch_queue
dataset_data_provider = slim.dataset_data_provider
#import argparse
flags = tf.app.flags

FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('train_crop_size', [513, 513],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', 'deeplab/datasets/pascal_voc_seg/tfrecord', 'Where the dataset reside.')


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
    
#    items=['image','image_name','height','width','labels_class']
#    for item in items:
#        print(data_provider.get([item]))
        
    image, height, width = data_provider.get(
      [common.IMAGE, common.HEIGHT, common.WIDTH])
    
    sess=tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    
    print(type(image))
    
    # Some datasets do not contain image_name.
    if common.IMAGE_NAME in data_provider.list_items():
        image_name, = data_provider.get([common.IMAGE_NAME])
    else:
        image_name = tf.constant('')

    label = None
    if FLAGS.train_split != common.TEST_SET:
        label, = data_provider.get([common.LABELS_CLASS])
    
    if label is not None:
        if label.shape.ndims == 2:
            label = tf.expand_dims(label, 2)
        elif label.shape.ndims == 3 and label.shape.dims[2] == 1:
            pass
        else:
            raise ValueError('Input label shape must be [height, width], or '
                             '[height, width, 1].')

        label.set_shape([None, None, 1])
    crop_size=FLAGS.train_crop_size
    original_image, image, label = input_preprocess.preprocess_image_and_label(
        image,
        label,
        crop_height=crop_size[0],
        crop_width=crop_size[1],
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        min_scale_factor=FLAGS.min_scale_factor,
        max_scale_factor=FLAGS.max_scale_factor,
        scale_factor_step_size=FLAGS.scale_factor_step_size,
        ignore_label=dataset.ignore_label,
        is_training=is_training,
        model_variant=FLAGS.model_variant)
    sample = {
        common.IMAGE: image,
        common.IMAGE_NAME: image_name,
        common.HEIGHT: height,
        common.WIDTH: width
    }
    if label is not None:
        sample[common.LABEL] = label
    
    num_threads = 1
    if not is_training:
        # Original image is only used during visualization.
        sample[common.ORIGINAL_IMAGE] = original_image,
        num_threads = 1
        
    batch_size=2
    
#    device='/device:GPU:0' if tf.test.is_gpu_available() else '/device:CPU:0'
    
    samples = tf.train.batch(
        sample,
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=2 * batch_size,
        allow_smaller_final_batch=not is_training,
        dynamic_pad=True)
    
    for key in samples.keys():
        print(key)
    
    num_clones=2
    inputs_queue = prefetch_queue.prefetch_queue(
          samples, capacity=2 * num_clones)
     
    samples = inputs_queue.dequeue()
    print('dequeue'+'*'*50)
    # Add name to input and label nodes so we can add to summary.
    samples[common.IMAGE] = tf.identity(
        samples[common.IMAGE], name=common.IMAGE)
    samples[common.LABEL] = tf.identity(
        samples[common.LABEL], name=common.LABEL)
    
    print('image shape',image.shape, type(image))
    print(samples[common.IMAGE].shape)
    print(samples[common.LABEL].shape)
    print(samples[common.IMAGE].dtype)
    print(samples[common.LABEL].dtype)
    tf.train.start_queue_runners(sess)
    image=sess.run(samples[common.IMAGE])
    label=sess.run(samples[common.LABEL])
    ids=np.unique(label)
    print('unique ids is: ',ids)
    print('end'+'.'*50)
    
    edge=batch_get_edge(label)
    samples[common.EDGE]=tf.convert_to_tensor(edge,dtype=tf.int32,name=common.EDGE)
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