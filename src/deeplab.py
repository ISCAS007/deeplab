# -*- coding: utf-8 -*-
from deeplab import common
from deeplab import model
import tensorflow as tf
from deeplab.utils import train_utils
import six

class deeplab():
    def __init__(self,flags):
        self.flags=flags
    
    def _build_model(self, inputs_queue, outputs_to_num_classes, ignore_label):
      """Builds a clone of DeepLab.
    
      Args:
        inputs_queue: A prefetch queue for images and labels.
        outputs_to_num_classes: A map from output type to the number of classes.
          For example, for the task of semantic segmentation with 21 semantic
          classes, we would have outputs_to_num_classes['semantic'] = 21.
        ignore_label: Ignore label.
    
      Returns:
        A map of maps from output_type (e.g., semantic prediction) to a
          dictionary of multi-scale logits names to logits. For each output_type,
          the dictionary has keys which correspond to the scales and values which
          correspond to the logits. For example, if `scales` equals [1.0, 1.5],
          then the keys would include 'merged_logits', 'logits_1.00' and
          'logits_1.50'.
      """
      FLAGS=self.flags
      samples = inputs_queue.dequeue()
    
      # Add name to input and label nodes so we can add to summary.
      samples[common.IMAGE] = tf.identity(
          samples[common.IMAGE], name=common.IMAGE)
      samples[common.LABEL] = tf.identity(
          samples[common.LABEL], name=common.LABEL)
    
      model_options = common.ModelOptions(
          outputs_to_num_classes=outputs_to_num_classes,
          crop_size=FLAGS.train_crop_size,
          atrous_rates=FLAGS.atrous_rates,
          output_stride=FLAGS.output_stride)
      outputs_to_scales_to_logits = model.multi_scale_logits(
          samples[common.IMAGE],
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid,
          weight_decay=FLAGS.weight_decay,
          is_training=True,
          fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)
    
      # Add name to graph node so we can add to summary.
      output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
      output_type_dict[model.MERGED_LOGITS_SCOPE] = tf.identity(
          output_type_dict[model.MERGED_LOGITS_SCOPE],
          name=common.OUTPUT_TYPE)
    
      for output, num_classes in six.iteritems(outputs_to_num_classes):
        train_utils.add_softmax_cross_entropy_loss_for_each_scale(
            outputs_to_scales_to_logits[output],
            samples[common.LABEL],
            num_classes,
            ignore_label,
            loss_weight=1.0,
            upsample_logits=FLAGS.upsample_logits,
            scope=output)
    
      return outputs_to_scales_to_logits
