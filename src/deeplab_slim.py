# -*- coding: utf-8 -*-

import tensorflow as tf
slim=tf.contrib.slim
from src.pspnet import pspnet,get_dataset
from deeplab import common,model
from deeplab.utils import train_utils
import os

class deeplab_slim(pspnet):
    def __init__(self,flags):
        super().__init__(flags)
    
    def train(self):
        FLAGS=self.flags        
        image_batch, annotation_batch = get_dataset(FLAGS,mode=tf.estimator.ModeKeys.TRAIN)
        
        outputs_to_num_classes={common.OUTPUT_TYPE: self.num_classes}
        model_options = common.ModelOptions(
            outputs_to_num_classes=outputs_to_num_classes,
            crop_size=FLAGS.train_crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)
        
        # outputs_to_scales_to_logits[key_1][key_2]=logits
        # key_1 in outputs_to_num_classes.keys()
        # key_2 in ['logits_%.2f' % image_scale for image_scale in image_pyramid]+[MERGED_LOGITS_SCOPE]
        outputs_to_scales_to_logits = model.multi_scale_logits(
            image_batch,
            model_options=model_options,
            image_pyramid=FLAGS.image_pyramid,
            weight_decay=FLAGS.weight_decay,
            is_training=True,
            fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)
    
        # Add name to graph node so we can add to summary.
        output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
        logits = output_type_dict[model.MERGED_LOGITS_SCOPE] = tf.identity(
            output_type_dict[model.MERGED_LOGITS_SCOPE],
            name=common.OUTPUT_TYPE)
        labels = annotation_batch
        
        if FLAGS.upsample_logits:
            # Label is not downsampled, and instead we upsample logits.
            logits = tf.image.resize_bilinear(
                logits,
                tf.shape(labels)[1:3],
                align_corners=True)
            scaled_labels = labels
        else:
            # Label is downsampled to the same size as logits.
            scaled_labels = tf.image.resize_nearest_neighbor(
                annotation_batch,
                tf.shape(logits)[1:3],
                align_corners=True)
        
        self.get_metric(scaled_labels,logits,'train')
        
        total_loss=0
        # outputs_to_scales_to_logits[output]={}
        for output, num_classes in outputs_to_num_classes.items():
            total_loss+=train_utils.add_softmax_cross_entropy_loss_for_each_scale(
                outputs_to_scales_to_logits[output],
                annotation_batch,
                num_classes,
                self.ignore_label,
                loss_weight=1.0,
                upsample_logits=FLAGS.upsample_logits,
                scope=output)
        total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
        tf.summary.scalar('losses/total_loss', total_loss)
        
        learning_rate = train_utils.get_model_learning_rate(
                    FLAGS.learning_policy, FLAGS.base_learning_rate,
                    FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
                    FLAGS.training_number_of_steps, FLAGS.learning_power,
                    FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, FLAGS.momentum)
        tf.summary.scalar('learning_rate', learning_rate)
        
        global_step = tf.train.get_or_create_global_step()
        train_tensor=optimizer.minimize(total_loss,global_step)
        summary_op=tf.summary.merge_all()
        
        session_config = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        
        init_fn=slim.assign_from_checkpoint_fn(model_path=FLAGS.tf_initial_checkpoint,
                                                   var_list=slim.get_variables(),
                                                   ignore_missing_vars=True)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(FLAGS.train_logdir)
        sess=tf.Session(config=session_config)
        init_fn(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess)
        
        for i in trange(FLAGS.training_number_of_steps):
            loss,summary=sess.run([train_tensor,summary_op])
            train_writer.add_summary(summary,i)
        
        saver.save(sess,os.path.join(FLAGS.train_logdir,'model'),global_step=FLAGS.training_number_of_steps)
        train_writer.close()