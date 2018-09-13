# -*- coding: utf-8 -*-
"""
run model
"""

import tensorflow as tf
from deeplab import common
from src.deeplab_edge import deeplab_edge
from src.deeplab_base import deeplab_base
from src.deeplab_global import deeplab_global
from src.utils import tf_config
import os

def main(unused_argv):
    flags=tf_config.FLAGS
    print(flags.flags_into_string())
    
    if flags.model_variant=='xception_65':
        if flags.tf_initial_checkpoint in [None,'xception']:
            flags.tf_initial_checkpoint='deeplab/datasets/weights/xception/model.ckpt'
        elif flags.tf_initial_checkpoint in ['pascal_train_aug','voc','pascal_voc']:
            flags.tf_initial_checkpoint='deeplab/datasets/weights/deeplabv3_pascal_train_aug/model.ckpt'
        
        flags.train_logdir=os.path.join(os.path.expanduser('~/tmp/logs/tensorflow'),flags.net_name,flags.dataset,flags.note)
        flags.checkpoint_dir=flags.train_logdir
        flags.eval_logdir=os.path.join(os.path.expanduser('~/tmp/logs/tensorflow'),flags.net_name,flags.dataset,flags.note,'eval')
        flags.dataset_dir='deeplab/datasets/cityscapes/tfrecord'
    else:
        assert False,'unknown model variant %s'%flags.model_variant
    
    net=globals()[flags.net_name](flags)
    
    if flags.app=='train':
        net.train()
    elif flags.app=='val':
        net.val()
    elif flags.app=='dump':
        net.dump()

if __name__ == '__main__':
  tf.app.run()
