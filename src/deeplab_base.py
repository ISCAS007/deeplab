# -*- coding: utf-8 -*-
from deeplab import common
#from deeplab import model
import tensorflow as tf
from deeplab.utils import train_utils
from deeplab.core import feature_extractor
from src.dataset.dataset_pipeline import get_dataset_files, dataset_pipeline, preprocess_image_and_label
from torch.utils import data as td
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from tqdm import trange,tqdm
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deployment import model_deploy
from deeplab import model
from src.pspnet import pspnet
#from deeplab.datasets import segmentation_dataset
import numpy as np
import time
import six
import math
import os

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'

DATASETS_CLASS_NUM = {
    'cityscapes': 19,
    'pascal_voc_seg': 21,
    'ade20k': 151,
}

DATASETS_IGNORE_LABEL = {
    'cityscapes': 255,
    'pascal_voc_seg': 255,
    'ade20k': 0,
}

class deeplab_base(pspnet):
    def __init__(self, flags):
        super().__init__(flags)
        print('num_classes',self.num_classes)
        print('ignore_label',self.ignore_label)
        #self.flags = flags
        # self.name = self.__class__.__name__
        
    def train(self):
        FLAGS = self.flags
        dataset_split = 'train'
        edge_width=20
        img_files, label_files = get_dataset_files(
            FLAGS.dataset, dataset_split)

        dataset=edict()
        dataset_pp=dataset_pipeline(edge_width,img_files,label_files,is_train=True)
        dataset.num_classes=DATASETS_CLASS_NUM[FLAGS.dataset]
        dataset.ignore_label=DATASETS_IGNORE_LABEL[FLAGS.dataset]
        dataset.num_samples=len(dataset_pp)
        
        tf.logging.set_verbosity(tf.logging.INFO)
        # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
        config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.num_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)
    
        # Split the batch across GPUs.
        assert FLAGS.train_batch_size % config.num_clones == 0, (
            'Training batch size not divisble by number of clones (GPUs).')
    
        clone_batch_size = FLAGS.train_batch_size // config.num_clones
    
        # Get dataset-dependent information.
#        dataset = segmentation_dataset.get_dataset(
#            FLAGS.dataset, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)
        

        tf.gfile.MakeDirs(FLAGS.train_logdir)
        tf.logging.info('Training on %s set', FLAGS.train_split)
    
        with tf.Graph().as_default() as graph:
            with tf.device(config.inputs_device()):
                data_list=dataset_pp.iterator()
                samples = input_generator.get(
                    (data_list,dataset.ignore_label),
                    FLAGS.train_crop_size,
                    clone_batch_size,
                    min_resize_value=FLAGS.min_resize_value,
                    max_resize_value=FLAGS.max_resize_value,
                    resize_factor=FLAGS.resize_factor,
                    min_scale_factor=FLAGS.min_scale_factor,
                    max_scale_factor=FLAGS.max_scale_factor,
                    scale_factor_step_size=FLAGS.scale_factor_step_size,
                    dataset_split=FLAGS.train_split,
                    is_training=True,
                    model_variant=FLAGS.model_variant)
                inputs_queue = prefetch_queue.prefetch_queue(
                    samples, capacity=128 * config.num_clones)
    
            # Create the global step on the device storing the variables.
            with tf.device(config.variables_device()):
                global_step = tf.train.get_or_create_global_step()
    
                # Define the model and create clones.
                model_fn = self._build_deeplab
                model_args = (inputs_queue, {
                    common.OUTPUT_TYPE: dataset.num_classes
                }, dataset.ignore_label)
                clones = model_deploy.create_clones(
                    config, model_fn, args=model_args)
    
                # Gather update_ops from the first clone. These contain, for example,
                # the updates for the batch_norm variables created by model_fn.
                first_clone_scope = config.clone_scope(0)
                update_ops = tf.get_collection(
                    tf.GraphKeys.UPDATE_OPS, first_clone_scope)
    
            # Gather initial summaries.
            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    
            # Add summaries for model variables.
            for model_var in slim.get_model_variables():
                summaries.add(tf.summary.histogram(model_var.op.name, model_var))
            
            label_name=('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/')
            print('first clone label name is:',label_name)
                
            # Add summaries for images, labels, semantic predictions
            if FLAGS.save_summaries_images:
                summary_image = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, common.IMAGE)).strip('/'))
                summaries.add(
                    tf.summary.image('samples/%s' % common.IMAGE, summary_image))
                
                first_clone_label = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/'))
                
                # Scale up summary image pixel values for better visualization.
                pixel_scaling = max(1, 255 // dataset.num_classes)
                summary_label = tf.cast(
                    first_clone_label * pixel_scaling, tf.uint8)
                summaries.add(
                    tf.summary.image('samples/%s' % common.LABEL, summary_label))
    
                first_clone_output = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, common.OUTPUT_TYPE)).strip('/'))
                predictions = tf.expand_dims(tf.argmax(first_clone_output, 3), -1)
    
                summary_predictions = tf.cast(
                    predictions * pixel_scaling, tf.uint8)
                summaries.add(
                    tf.summary.image(
                        'samples/%s' % common.OUTPUT_TYPE, summary_predictions))
            
            # Add summaries for miou,acc
            labels = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/'))
            predictions = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, common.OUTPUT_TYPE)).strip('/'))
            predictions = tf.image.resize_bilinear(predictions,tf.shape(labels)[1:3],align_corners=True)
            # predictions shape (2, 513, 513, 19/21)
            print('predictions shape',predictions.shape)
            self.get_metric(labels,predictions,'train')
            
            # Add summaries for losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
                summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
            
#            losses = {}
#            for key in [common.OUTPUT_TYPE,common.EDGE]:
#                losses[key]=graph.get_tensor_by_name(name='losses/%s:0'%key)
#                summaries.add(tf.summary.scalar('losses/'+key,losses[key]))
                
            # Build the optimizer based on the device specification.
            with tf.device(config.optimizer_device()):
                learning_rate = train_utils.get_model_learning_rate(
                    FLAGS.learning_policy, FLAGS.base_learning_rate,
                    FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
                    FLAGS.training_number_of_steps, FLAGS.learning_power,
                    FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate, FLAGS.momentum)
                summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    
            startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps
            for variable in slim.get_model_variables():
                summaries.add(tf.summary.histogram(variable.op.name, variable))
    
            with tf.device(config.variables_device()):
                total_loss, grads_and_vars = model_deploy.optimize_clones(
                    clones, optimizer)
                total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
                summaries.add(tf.summary.scalar('losses/total_loss', total_loss))
    
                # Modify the gradients for biases and last layer variables.
                last_layers = model.get_extra_layer_scopes(
                    FLAGS.last_layers_contain_logits_only)
                grad_mult = train_utils.get_model_gradient_multipliers(
                    last_layers, FLAGS.last_layer_gradient_multiplier)
                if grad_mult:
                    grads_and_vars = slim.learning.multiply_gradients(
                        grads_and_vars, grad_mult)
    
                # Create gradient update op.
                grad_updates = optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)
                update_ops.append(grad_updates)
                update_op = tf.group(*update_ops)
                with tf.control_dependencies([update_op]):
                    train_tensor = tf.identity(total_loss, name='train_op')
    
            # Add the summaries from the first clone. These contain the summaries
            # created by model_fn and either optimize_clones() or _gather_clone_loss().
            summaries |= set(
                tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
    
            # Merge all summaries together.
            summary_op = tf.summary.merge(list(summaries))
    
            # Soft placement allows placing on CPU ops without GPU implementation.
            session_config = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)
            session_config.gpu_options.allow_growth = True
            
            init_fn=train_utils.get_model_init_fn(
                    FLAGS.train_logdir,
                    FLAGS.tf_initial_checkpoint,
                    FLAGS.initialize_last_layer,
                    last_layers,
                    ignore_missing_vars=True)
#            init_fn=slim.assign_from_checkpoint_fn(model_path=FLAGS.tf_initial_checkpoint,
#                                                   var_list=slim.get_variables(),
#                                                   ignore_missing_vars=True)
#            saver = tf.train.Saver()
#            train_writer = tf.summary.FileWriter(FLAGS.train_logdir)
#            sess=tf.Session(config=session_config)
#            init_fn(sess)
#            sess.run(tf.global_variables_initializer())
#            sess.run(tf.local_variables_initializer())
#            tf.train.start_queue_runners(sess)
#            
#            for i in trange(FLAGS.training_number_of_steps):
#                loss,summary=sess.run([train_tensor,summary_op])
#                train_writer.add_summary(summary,i)
#            
#            saver.save(sess,os.path.join(FLAGS.train_logdir,'model'),global_step=FLAGS.training_number_of_steps)
#            train_writer.close()
            # Start the training.
            slim.learning.train(
                train_tensor,
                logdir=FLAGS.train_logdir,
                log_every_n_steps=FLAGS.log_steps,
                master=FLAGS.master,
                number_of_steps=FLAGS.training_number_of_steps,
                is_chief=(FLAGS.task == 0),
                session_config=session_config,
                startup_delay_steps=startup_delay_steps,
                init_fn=init_fn,
                summary_op=summary_op,
                save_summaries_secs=FLAGS.save_summaries_secs,
                save_interval_secs=FLAGS.save_interval_secs)

    def val(self):
        FLAGS=self.flags
        tf.logging.set_verbosity(tf.logging.INFO)
        # Get dataset-dependent information.
#        dataset = segmentation_dataset.get_dataset(
#            FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)
        dataset_split='val'
        edge_width=20
        img_files,label_files=get_dataset_files(FLAGS.dataset,dataset_split)
        dataset_pp=dataset_pipeline(edge_width,img_files,label_files,is_train=False)
        num_classes=DATASETS_CLASS_NUM[FLAGS.dataset]
        ignore_label=DATASETS_IGNORE_LABEL[FLAGS.dataset]
        num_samples=len(dataset_pp)
        
        log_dir = os.path.join(os.path.expanduser(
            '~/tmp/logs/tensorflow'), self.flags.net_name, self.flags.dataset, 'eval')
#        os.makedirs(log_dir, exist_ok=True)
        FLAGS.eval_logdir=log_dir
        print('eval_logdir is',log_dir)
        print('checkpoint dir is',FLAGS.checkpoint_dir)
        
        tf.gfile.MakeDirs(FLAGS.eval_logdir)
        tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

        with tf.Graph().as_default():
            data_list=dataset_pp.iterator()
            samples = input_generator.get(
                (data_list,ignore_label),
                FLAGS.eval_crop_size,
                FLAGS.eval_batch_size,
                min_resize_value=FLAGS.min_resize_value,
                max_resize_value=FLAGS.max_resize_value,
                resize_factor=FLAGS.resize_factor,
                dataset_split=FLAGS.eval_split,
                is_training=False,
                model_variant=FLAGS.model_variant)

            model_options = common.ModelOptions(
                outputs_to_num_classes={
                    common.OUTPUT_TYPE: num_classes},
                crop_size=FLAGS.eval_crop_size,
                atrous_rates=FLAGS.atrous_rates,
                output_stride=FLAGS.output_stride)

            if tuple(FLAGS.eval_scales) == (1.0,):
                tf.logging.info('Performing single-scale test.')
                predictions = model.predict_labels(samples[common.IMAGE], model_options,
                                                   image_pyramid=FLAGS.image_pyramid)
            else:
                tf.logging.info('Performing multi-scale test.')
                predictions = model.predict_labels_multi_scale(
                    samples[common.IMAGE],
                    model_options=model_options,
                    eval_scales=FLAGS.eval_scales,
                    add_flipped_images=FLAGS.add_flipped_images)
            predictions = predictions[common.OUTPUT_TYPE]
            predictions = tf.reshape(predictions, shape=[-1])
            labels = tf.reshape(samples[common.LABEL], shape=[-1])
            weights = tf.to_float(tf.not_equal(labels, ignore_label))

            # Set ignore_label regions to label 0, because metrics.mean_iou requires
            # range of labels = [0, dataset.num_classes). Note the ignore_label regions
            # are not evaluated since the corresponding regions contain weights = 0.
            labels = tf.where(
                tf.equal(labels, ignore_label), tf.zeros_like(labels), labels)

            predictions_tag = 'miou'
            for eval_scale in FLAGS.eval_scales:
                predictions_tag += '_' + str(eval_scale)
            if FLAGS.add_flipped_images:
                predictions_tag += '_flipped'

            # Define the evaluation metric.
            metric_map = {}
            metric_map[predictions_tag] = tf.metrics.mean_iou(
                predictions, labels, num_classes, weights=weights)

            metrics_to_values, metrics_to_updates = (
                tf.contrib.metrics.aggregate_metric_map(metric_map))

            for metric_name, metric_value in six.iteritems(metrics_to_values):
                slim.summaries.add_scalar_summary(
                    metric_value, metric_name, print_summary=True)

            num_batches = int(
                math.ceil(num_samples / float(FLAGS.eval_batch_size)))

            tf.logging.info('Eval num images %d', num_samples)
            tf.logging.info('Eval batch size %d and num batch %d',
                            FLAGS.eval_batch_size, num_batches)

            num_eval_iters = None
            if FLAGS.max_number_of_evaluations > 0:
                num_eval_iters = FLAGS.max_number_of_evaluations
            slim.evaluation.evaluation_loop(
                master=FLAGS.master,
                checkpoint_dir=FLAGS.checkpoint_dir,
                logdir=FLAGS.eval_logdir,
                num_evals=num_batches,
                eval_op=list(metrics_to_updates.values()),
                max_number_of_evaluations=num_eval_iters,
                eval_interval_secs=FLAGS.eval_interval_secs)

    def dump(self):
        pass

    def _build_deeplab(self, inputs_queue, outputs_to_num_classes, ignore_label):
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
