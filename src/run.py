# -*- coding: utf-8 -*-
"""
run model
"""

import tensorflow as tf
from deeplab import common
from src.utils import config
import os

def main(unused_argv):
    flags=config.FLAGS
    print(flags.flags_into_string())
    
    if flags.model_variant=='xception_65':
        if flags.tf_initial_checkpoint is None:
            flags.tf_initial_checkpoint='deeplab/datasets/weights/xception/model.ckpt'
        flags.train_logdir=os.path.expanduser('~/tmp/logs/tensorflow')
        flags.dataset_dir=None
    else:
        assert False,'unknown model variant %s'%flags.model_variant
        

if __name__ == '__main__':
  tf.app.run()
