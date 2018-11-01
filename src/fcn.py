# -*- coding: utf-8 -*-

import sys
if '.' not in sys.path:
    sys.path.append('.')
import os
# sys.path.append(os.path.expanduser('~/git/gnu/tf-image-segmentation'))
from tf_image_segmentation.models.fcn_32s import FCN_32s, extract_vgg_16_mapping_without_fc8
from tf_image_segmentation.utils.training import get_valid_logits_and_labels
from src.dataset.dataset_pipeline import DATASETS_CLASS_NUM, DATASETS_IGNORE_LABEL, get_dataset_files, dataset_pipeline, preprocess_image_and_label
import tensorflow as tf
from src.pspnet import get_dataset, pspnet
from tqdm import tqdm,trange
slim=tf.contrib.slim

class fcn(pspnet):
    def __init__(self,flags):
        self.flags = flags
        self.num_classes = DATASETS_CLASS_NUM[self.flags.dataset]
        self.ignore_label = DATASETS_IGNORE_LABEL[self.flags.dataset]
        # flags.DEFINE_integer('output_stride', 16, 'The ratio of input to output spatial resolution.')
        self.output_stride = self.flags.output_stride
        # flags.DEFINE_string('model_variant', 'mobilenet_v2', 'DeepLab model variant.')
        model_variant = self.flags.model_variant
        self.model = None

        if model_variant.lower().startswith('xception'):
            self.backbone_name = 'xception'
        else:
            self.backbone_name = model_variant
            
    def train(self):
        FLAGS=self.flags        
        image_batch, annotation_batch = get_dataset(FLAGS,mode=tf.estimator.ModeKeys.TRAIN)
        upsampled_logits_batch, vgg_16_variables_mapping = FCN_32s(image_batch_tensor=image_batch,
                                                                   number_of_classes=DATASETS_CLASS_NUM[FLAGS.dataset],
                                                                   is_training=True)
        
        loss=self.get_loss(labels=annotation_batch,logits=upsampled_logits_batch)
        self.get_metric(labels=annotation_batch, logits=upsampled_logits_batch,mode_str='train')
        train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.base_learning_rate).minimize(loss,  
                                           tf.train.get_or_create_global_step())


        # Variable's initialization functions
        vgg_16_without_fc8_variables_mapping = extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping)
        
        
        init_fn = slim.assign_from_checkpoint_fn(model_path=FLAGS.tf_initial_checkpoint,
                                                 var_list=vgg_16_without_fc8_variables_mapping)
        
        tf.summary.scalar('cross_entropy_loss', loss)
        
        summary_op=tf.summary.merge_all()
        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(FLAGS.train_logdir)
        sess=tf.Session(config=session_config)
        init_fn(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess)
        
        for i in trange(FLAGS.training_number_of_steps):
            loss,summary=sess.run([train_step,summary_op])
            train_writer.add_summary(summary,i)
        
        saver.save(sess,os.path.join(FLAGS.train_logdir,'model'),global_step=FLAGS.training_number_of_steps)
        train_writer.close()
#        startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps
#        slim.learning.train(train_step,
#                            logdir=FLAGS.train_logdir,
#                            log_every_n_steps=FLAGS.log_steps,
#                            master=FLAGS.master,
#                            number_of_steps=FLAGS.training_number_of_steps,
#                            is_chief=(FLAGS.task == 0),
#                            session_config=session_config,
#                            startup_delay_steps=startup_delay_steps,
#                            init_fn=init_fn,
#                            summary_op=summary_op,
#                            save_summaries_secs=FLAGS.save_summaries_secs,
#                            save_interval_secs=FLAGS.save_interval_secs)
    def val(self):
        pass
    
    def dump(self):
        pass