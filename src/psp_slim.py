# -*- coding: utf-8 -*-

import sys
if '.' not in sys.path:
    sys.path.append('.')
import tensorflow as tf
import os
from tqdm import tqdm

from easydict import EasyDict as edict
from src.dataset.dataset_pipeline import DATASETS_CLASS_NUM, DATASETS_IGNORE_LABEL, get_dataset_files, dataset_pipeline, preprocess_image_and_label
from deeplab.utils import input_generator
from deeplab import common
from src.utils import tf_config
from src.pspnet import get_dataset, pspnet
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
slim=tf.contrib.slim

class psp_slim(pspnet):
    def __init__(self,flags):
        self.flags=flags
        self.num_classes = DATASETS_CLASS_NUM[self.flags.dataset]
        self.ignore_label = DATASETS_IGNORE_LABEL[self.flags.dataset]
        # flags.DEFINE_integer('output_stride', 16, 'The ratio of input to output spatial resolution.')
        self.output_stride = self.flags.output_stride
        
    def get_model(self,features,labels):
        pass
    
    def get_backbone(self,features):
        if self.flags.model_variant.startswith('xception'):
            assert False,'not implement'
        elif self.flags.model_variant=='resnet_v2_50':
            # inputs has shape [batch, 513, 513, 3]
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                net, end_points = resnet_v2.resnet_v2_50(features,
                                                self.num_classes,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=self.output_stride)
        elif self.flags.model_variant=='resnet_v1_50':
            # The key difference of the full preactivation 'v2' variant compared to the
            # 'v1' variant in [1] is the use of batch normalization before every weight layer.
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = resnet_v1.resnet_v1_50(features,
                                                self.num_classes,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=self.output_stride)
        elif self.flags.model_variant=='resnet_v2_101':
            # inputs has shape [batch, 513, 513, 3]
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                net, end_points = resnet_v2.resnet_v2_101(features,
                                                self.num_classes,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=self.output_stride)
        elif self.flags.model_variant=='resnet_v1_101':
            # The key difference of the full preactivation 'v2' variant compared to the
            # 'v1' variant in [1] is the use of batch normalization before every weight layer.
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = resnet_v1.resnet_v1_101(features,
                                                self.num_classes,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=self.output_stride)
        else:
            assert False,'not implement'
            
        print(end_points.keys())
        print(net)
        
    def train(self):
        FLAGS=self.flags        
        image_batch, annotation_batch = get_dataset(FLAGS,mode=tf.estimator.ModeKeys.TRAIN)
        model=self.get_model(image_batch,annotation_batch)