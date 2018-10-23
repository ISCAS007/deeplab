# -*- coding: utf-8 -*-

import tensorflow as tf
tl=tf.keras.layers
from src.utils.disc import lcm_list
    
class resize_bilinear_layer(tl.Layer):
    def __init__(self,output_shape):
        super().__init__()
        assert len(output_shape)==2
        self.output_size=output_shape
    
    def compute_output_shape(self,input_shape):
        return(input_shape[0],self.output_size[0],self.output_size[1],input_shape[3])
        
    def call(self,inputs):
        return tf.image.resize_bilinear(inputs,self.output_size,align_corners=True)
    
    def get_config(self):
        config = {
                  'output_size': self.output_size,
                 }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def paramid_pooling_path(x,num_filters,stride):
    # data_format=channels_last default
    pp_path=tf.keras.Sequential([
            tl.AvgPool2D(pool_size=stride,strides=stride),
            tl.Conv2D(filters=num_filters,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      use_bias=False),
            tl.BatchNormalization(),
            tl.Activation('relu'),
            resize_bilinear_layer(x.shape[1:3])
            ])
    
    y=pp_path(x)
    return y,pp_path
    
def pyramid_pooling_module(x,num_filters=512,levels=[6,3,2,1],scale=15):
    input_size=lcm_list(levels)*scale
#    assert x.shape[1]==x.shape[2]==input_size,'input size %d not equal to %s'%(input_size,x.shape)
    
    pyramid_pooling_blocks = [x]
    pp_pathes=[]
    for level in levels:
        y,pp_path=paramid_pooling_path(x,num_filters,input_size//level)
        pp_pathes.append(pp_path)
        pyramid_pooling_blocks.append(y)
        
    y=tl.Concatenate()(pyramid_pooling_blocks)
    return y,pp_pathes