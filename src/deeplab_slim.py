# -*- coding: utf-8 -*-

import tensorflow as tf
slim=tf.contrib.slim
from src.pspnet import pspnet,get_dataset
from deeplab import common,model
from deeplab.utils import train_utils
import os
from tqdm import trange

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
        
        softmax_loss=0
        # outputs_to_scales_to_logits[output]={}
        for output, num_classes in outputs_to_num_classes.items():
            softmax_loss+=train_utils.add_softmax_cross_entropy_loss_for_each_scale(
                outputs_to_scales_to_logits[output],
                annotation_batch,
                num_classes,
                self.ignore_label,
                loss_weight=1.0,
                upsample_logits=FLAGS.upsample_logits,
                scope=output)
        
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss=tf.add_n(regularization_losses)
        tf.summary.scalar('losses/reg_loss', reg_loss)
        model_losses = tf.get_collection(tf.GraphKeys.LOSSES)
        model_loss=tf.add_n(model_losses)
        tf.summary.scalar('losses/model_loss', model_loss)
        
        learning_rate = train_utils.get_model_learning_rate(
                    FLAGS.learning_policy, FLAGS.base_learning_rate,
                    FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
                    FLAGS.training_number_of_steps, FLAGS.learning_power,
                    FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
            
        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        tf.summary.scalar('learning_rate', learning_rate)
        
        with tf.control_dependencies([tf.assert_equal(softmax_loss,model_loss)]):
            total_loss=model_loss+reg_loss
            total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
            tf.summary.scalar('losses/total_loss', total_loss)
        
        global_step = tf.train.get_or_create_global_step()
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        
#        train_tensor=optimizer.minimize(total_loss,global_step)
#        train_tensor=slim.learning.create_train_op(total_loss=total_loss,
#            optimizer=optimizer,
#            global_step=global_step)
        
        #BUG update the weight twice???
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')
        
        summary_op=tf.summary.merge_all()
        
        session_config = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        
        last_layers = model.get_extra_layer_scopes(
                    FLAGS.last_layers_contain_logits_only)
        exclude_list = ['global_step']
        if not FLAGS.initialize_last_layer:
            exclude_list.extend(last_layers)
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)
        init_fn=slim.assign_from_checkpoint_fn(model_path=FLAGS.tf_initial_checkpoint,
                                                   var_list=variables_to_restore,
                                                   ignore_missing_vars=True)
        
        #use the train_tensor with slim.learning.train, not session
#        saver = tf.train.Saver()
#        train_writer = tf.summary.FileWriter(FLAGS.train_logdir)
#        sess=tf.Session(config=session_config)
#        init_fn(sess)
#        sess.run(tf.global_variables_initializer())
#        sess.run(tf.local_variables_initializer())
#        sess.run(tf.tables_initializer())
#        tf.train.start_queue_runners(sess)
#        
#        for i in trange(FLAGS.training_number_of_steps):
#            loss,summary,n_step=sess.run([train_tensor,summary_op,global_step])
#            train_writer.add_summary(summary,i)
#            if i%100==1:
#                print('%d/%d global_step=%0.2f, loss='%(i,FLAGS.training_number_of_steps,n_step),loss)
#        
#        saver.save(sess,os.path.join(FLAGS.train_logdir,'model'),global_step=FLAGS.training_number_of_steps)
#        train_writer.close()
        
#        Start the training.
        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_logdir,
            log_every_n_steps=FLAGS.log_steps,
            master=FLAGS.master,
            is_chief=(FLAGS.task == 0),
            number_of_steps=FLAGS.training_number_of_steps,
            session_config=session_config,
            startup_delay_steps=0,
            init_fn=init_fn,
            summary_op=summary_op,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs)