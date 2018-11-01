# -*- coding: utf-8 -*-
import sys
if '.' not in sys.path:
    sys.path.append('.')
import tensorflow as tf
tl = tf.keras.layers
VGG16 = tf.keras.applications.VGG16
VGG19 = tf.keras.applications.VGG19
RESNET50 = ResNet50 = tf.keras.applications.ResNet50
XCEPTION = Xception = tf.keras.applications.Xception

from tqdm import tqdm
from easydict import EasyDict as edict
from src.dataset.dataset_pipeline import DATASETS_CLASS_NUM, DATASETS_IGNORE_LABEL, get_dataset_files, dataset_pipeline, preprocess_image_and_label
from deeplab.utils import input_generator
from deeplab import common
from src.utils import tf_config
import os
from src.utils.disc import get_backbone_index
import math
from src.tf_layers import pyramid_pooling_module, resize_bilinear_layer
import glob
from tensorflow.python.tools import inspect_checkpoint as chkp
slim=tf.contrib.slim


class pspnet(tf.keras.Model):
    def __init__(self, flags):
        super().__init__()
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

        self.train_crop_size = self.flags.train_crop_size
        self.eval_crop_size = self.flags.eval_crop_size

    def call(self, inputs, training=True):
        if training:
            pass
        else:
            pass
        
    def train(self):
        self.get_model(mode=tf.estimator.ModeKeys.TRAIN)
        print(tf.GraphKeys.TRAINABLE_VARIABLES)
        print('trainable',len(tf.trainable_variables()))
        print('local',len(tf.local_variables()))
        print('global',len(tf.global_variables()))
        print('model',len(tf.model_variables()))
        variables_to_restore = slim.get_model_variables()
        print('variable to restore',len(variables_to_restore))
        for v in tf.trainable_variables():
            print(v)
        
        model_dir = os.path.join(os.path.expanduser(
        '~/tmp/logs/tensorflow'), 'pspnet', self.flags.dataset, self.flags.note)
        checkpoint_file=tf.train.latest_checkpoint(model_dir)
        chkp.print_tensors_in_checkpoint_file(checkpoint_file,tensor_name=None,all_tensors=False)

    def get_backbone_weight(self):
        model_dir = os.path.join(os.path.expanduser(
            '~/tmp/logs/tensorflow'), 'pspnet', self.flags.dataset, self.flags.note)
        files = glob.glob(os.path.join(
            model_dir, 'model.ckpt*'), recursive=False)
        if len(files) == 0:
            return 'imagenet'
        else:
            return None

    def get_model(self, mode):
        #        if self.model is not None:
        #            return self.model
        if mode == tf.estimator.ModeKeys.TRAIN:
            output_size = self.train_crop_size
        else:
            output_size = self.eval_crop_size
        input_shape = tuple(output_size+[3])

        backbone_weight = self.get_backbone_weight()
        if backbone_weight is None:
            print('not load backbone weight', '-'*100)
        else:
            print('load backbone weight', '+'*100)
        # the first time we can load weight, however, when restore from checkpoing, load imagenet weight will cause error
        backbone = globals()[self.backbone_name.upper()](
            include_top=False, weights=backbone_weight, input_tensor=None, input_shape=input_shape, pooling=None, classes=1000)
        print('load backbone weight finished', '*'*100)

        backbone_index = get_backbone_index(
            self.backbone_name, int(math.log2(self.output_stride)))
        print('backbone_index',backbone_index)
#        print('backbone index is',backbone_index,len(self.backbone.layers))
        # x=self.backbone.get_output_at(backbone_index)
        x = backbone.layers[backbone_index].output
#        print('input feature shape',features.shape)
#        print('backbone feature shape',x.shape)
        x, midnets = pyramid_pooling_module(x, scale=5)
#        x=tf.image.resize_bilinear(x,features.shape[1:3],align_corners=True)
        suffix_net = tf.keras.Sequential([
            tl.Conv2D(filters=512,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      use_bias=False),
            tl.BatchNormalization(),
            tl.Activation('relu'),
            tl.Conv2D(filters=self.num_classes,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation='softmax'),
        ])
        x = suffix_net(x)
        x = resize_bilinear_layer(output_size)(x)
        model = tf.keras.Model(inputs=backbone.inputs, outputs=x)
        return model

    def get_loss(self, labels, logits):
        reshape_labels = tf.reshape(labels, shape=[-1])
        not_ignore_mask = tf.to_float(tf.not_equal(reshape_labels,
                                                   self.ignore_label))
        one_hot_labels = tf.one_hot(
            reshape_labels, self.num_classes, on_value=1.0, off_value=0.0)
        loss = tf.losses.softmax_cross_entropy(
            one_hot_labels,
            tf.reshape(logits, shape=[-1, self.num_classes]),
            weights=not_ignore_mask)

        return loss

    def get_metric(self, in_labels, in_logits, mode_str):
        """
        labels: b,h,w,1
        logits: b,h,w,c
        """
        labels = tf.reshape(in_labels, shape=[-1])
        logits = tf.reshape(tf.argmax(in_logits, -1), shape=[-1])

        weights = tf.to_float(tf.not_equal(labels, self.ignore_label))
        labels = tf.where(
            tf.equal(labels, self.ignore_label), tf.zeros_like(labels), labels)
        
        labels=tf.to_int32(labels)
        logits=tf.to_int32(logits)
        tf_num_classes=tf.to_int32(self.num_classes)
        with tf.control_dependencies([tf.assert_less(labels, tf_num_classes),tf.assert_less(logits,tf_num_classes)]):
            metric_map = {}
            metric_map['miou'] = tf.metrics.mean_iou(
                labels=labels, predictions=logits, num_classes=self.num_classes, weights=weights)
            metric_map['acc'] = tf.metrics.accuracy(
                labels=labels, predictions=logits, weights=tf.reshape(weights, shape=[-1]))
    
            for x in ['miou', 'acc']:
                tf.identity(metric_map[x][0], name='%s/%s' % (mode_str, x))
                op=tf.summary.scalar('%s/%s' % (mode_str, x), metric_map[x][0])
                tf.Print(op,[metric_map[x][0]],'%s/%s' % (mode_str, x))
                
                tf.identity(metric_map[x][1], name='%s/update_%s' % (mode_str, x))
                tf.summary.scalar('%s/update_%s' % (mode_str, x),
                                  tf.reduce_mean(metric_map[x][1]))
    
            hooks = [
                tf.train.LoggingTensorHook(
                    ['%s/%s' % (mode_str, x) for x in ['miou', 'acc']],
                    every_n_iter=100)
            ]
    
            return metric_map, hooks

    def model_function(self, features, labels, mode):
        """
        train mode and val mode model
        note:
            summary op will run only for train mode model
        """
        FLAGS = self.flags
        model = self.get_model(mode)

#        model.summary()
        if mode == tf.estimator.ModeKeys.TRAIN:
            logits = model(features)

            loss = self.get_loss(labels, logits)
#            print(loss.shape,loss)
            tf.identity(loss, 'train/loss')
            tf.summary.scalar('train/loss', loss)

            eval_metric_ops, hooks = self.get_metric(labels, logits, 'train')

            optimizer = tf.train.AdamOptimizer(
                learning_rate=FLAGS.base_learning_rate)
            lr = tf.identity(FLAGS.base_learning_rate, name='learning_rate')
            tf.summary.scalar('train/learning_rate', lr)

            labels = tf.where(tf.equal(labels, self.ignore_label),
                              tf.zeros_like(labels), labels)
            tf.summary.image(name='labels', tensor=tf.cast(
                labels*(255//self.num_classes), tf.uint8), max_outputs=20)
            tf.summary.image(name='outputs', tensor=tf.cast(tf.expand_dims(
                tf.argmax(logits, -1), -1)*(255//self.num_classes), tf.uint8), max_outputs=20)

            # create an estimator spec to optimize the loss
            estimator_spec = tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=optimizer.minimize(
                    loss, tf.train.get_or_create_global_step()),
                eval_metric_ops=eval_metric_ops,
                training_hooks=hooks)

        elif mode == tf.estimator.ModeKeys.EVAL:
            # pass the input through the model
            logits = model(features, training=False)

            loss = self.get_loss(labels, logits)
            tf.identity(loss, name='val_loss')
            tf.summary.scalar('val/loss', loss)

            # evalution will not write summary in self.get_metric like train
            # use eval_metric_ops instead
            eval_metric_ops, hooks = self.get_metric(labels, logits, 'val')

            # create an estimator spec with the loss and accuracy
            estimator_spec = tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=hooks,
            )
        else:
            assert False, 'unknown mode %s' % mode

        return estimator_spec

    def input_fn(self):
        if self.flags.app == 'train':
            mode = tf.estimator.ModeKeys.TRAIN
        else:
            mode = tf.estimator.ModeKeys.EVAL
        return get_dataset(self.flags, mode)

    def train_input_fn(self):
        return get_dataset(self.flags, tf.estimator.ModeKeys.TRAIN)

    def eval_input_fn(self):
        return get_dataset(self.flags, tf.estimator.ModeKeys.EVAL)


def get_dataset(flags, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset_split = 'train'
    elif mode == tf.estimator.ModeKeys.EVAL:
        dataset_split = 'val'
    else:
        assert False, 'unknown mode'

    FLAGS = flags
    data_config = edict()
    data_config.edge_width = 20
    data_config.ignore_label = DATASETS_IGNORE_LABEL[FLAGS.dataset]
    data_config.edge_class_num = FLAGS.edge_class_num
    img_files, label_files = get_dataset_files(
        FLAGS.dataset, dataset_split)

    dataset_pp = dataset_pipeline(
        data_config, img_files, label_files, is_train=True)
    data_list = dataset_pp.iterator()

    if mode == tf.estimator.ModeKeys.TRAIN:
        samples = input_generator.get(
            (data_list, data_config.ignore_label),
            FLAGS.train_crop_size,
            FLAGS.train_batch_size//FLAGS.num_clones,
            min_resize_value=FLAGS.min_resize_value,
            max_resize_value=FLAGS.max_resize_value,
            resize_factor=FLAGS.resize_factor,
            min_scale_factor=FLAGS.min_scale_factor,
            max_scale_factor=FLAGS.max_scale_factor,
            scale_factor_step_size=FLAGS.scale_factor_step_size,
            dataset_split=FLAGS.train_split,
            is_training=True,
            model_variant=FLAGS.model_variant)
    elif mode == tf.estimator.ModeKeys.EVAL:
        samples = input_generator.get(
            (data_list, data_config.ignore_label),
            FLAGS.eval_crop_size,
            FLAGS.eval_batch_size,
            min_resize_value=FLAGS.min_resize_value,
            max_resize_value=FLAGS.max_resize_value,
            resize_factor=FLAGS.resize_factor,
            dataset_split=FLAGS.eval_split,
            is_training=False,
            model_variant=FLAGS.model_variant)
    else:
        assert False, 'unknown mode'

    return samples[common.IMAGE], samples[common.LABEL]


def main(unused_argv):
    flags = tf_config.FLAGS
    img_files, label_files = get_dataset_files(
        flags.dataset, 'train')
    train_epoch_steps = len(img_files)
    img_files, label_files = get_dataset_files(
        flags.dataset, 'val')
    val_epoch_steps = len(img_files)

    model_dir = os.path.join(os.path.expanduser(
        '~/tmp/logs/tensorflow'), 'pspnet', flags.dataset, flags.note)

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    config = tf.estimator.RunConfig(
        model_dir=model_dir,
        tf_random_seed=None,
        save_summary_steps=train_epoch_steps,
        #            save_checkpoints_steps=_USE_DEFAULT,
        #            save_checkpoints_secs=_USE_DEFAULT,
        session_config=session_config,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=100)

    model = pspnet(flags)
    
    classifier = tf.estimator.Estimator(
        model_fn=model.model_function,
        model_dir=model_dir,
        config=config,
        params=None,
        warm_start_from=None)

    classifier.train(
        input_fn=model.train_input_fn,
        steps=flags.training_number_of_steps,
    )
    
    train_result=classifier.evaluate(
        input_fn=model.train_input_fn,
        steps=train_epoch_steps,
        name='train')
    print('train result',train_result)
    
    val_result=classifier.evaluate(
        input_fn=model.eval_input_fn,
        steps=val_epoch_steps,
        name='val')
    
    print('val result',val_result)

#    train_spec = tf.estimator.TrainSpec(input_fn=model.train_input_fn, max_steps=flags.training_number_of_steps)
#    eval_spec = tf.estimator.EvalSpec(input_fn=model.eval_input_fn, steps=20)
#
#    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
#    steps_per_eval=5
#    n_epoch=1+flags.training_number_of_steps//(steps_per_eval*train_epoch_steps)
#    for epoch in range(n_epoch):
#        train_classifier.train(
#            input_fn=train_model.train_input_fn,
#            steps=steps_per_eval*train_epoch_steps,
#            hooks=None,
#        )
#
#        train_classifier.evaluate(
#            input_fn=train_model.train_input_fn,
#            steps=train_epoch_steps,
#            hooks=None,
#            name='train'
#        )
#        train_classifier.evaluate(
#            input_fn=train_model.eval_input_fn,
#            steps=val_epoch_steps,
#            hooks=None,
#            name='val'
#        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
