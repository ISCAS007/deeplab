# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
VGG16=tf.keras.applications.VGG16
VGG19=tf.keras.applications.VGG19
RESNET50=ResNet50=tf.keras.applications.ResNet50
XCEPTION=Xception=tf.keras.applications.Xception
import math

# Minimum common multiple or least common multiple
def lcm(a,b):
    return a*b//math.gcd(a,b)

def lcm_list(l):
    x=1
    for i in l:
       x=lcm(i,x)
       
    return x

def get_edge(ann_img, edge_width=5,edge_class_num=2,ignore_index=None):
    kernel = np.ones((edge_width, edge_width), np.uint8)
    ann_edge = cv2.Canny(ann_img, 0, 1)
    if ignore_index is not None:
        # remove ignore area in ann_img
        ann_edge[ann_img==ignore_index]=0
        
    ann_dilation = cv2.dilate(ann_edge, kernel, iterations=1)
    
    if edge_class_num==2:
        # fg=0, bg=1
        edge_label = (ann_dilation == 0).astype(np.uint8)
    else:
        # fg=0, bg=1,2,...,edge_class_num-1
        edge_label = np.zeros_like(ann_img)+edge_class_num-1
        for class_num in range(edge_class_num-1):
            edge_label[np.logical_and(ann_dilation>0,edge_label==(edge_class_num-1))]=class_num
            ann_dilation = cv2.dilate(ann_dilation, kernel, iterations=1)
    
    if ignore_index is not None:
        edge_label[ann_img==ignore_index]=ignore_index
        
    return edge_label

def batch_get_edge(batch_ann_img,edge_width=5):
    b,h,w,c=batch_ann_img.shape
    assert c==1, 'the channel of ann image %d != 1'%c
    batch_edge=np.zeros_like(batch_ann_img)
    for idx in range(b):
        batch_edge[idx,:,:,0]=get_edge(batch_ann_img[idx,:,:,0].astype(np.uint8),edge_width)
    
    return batch_edge

def tf_batch_get_edge(session,tf_labels,name,edge_width=5):
    batch_ann_img=session.run(tf_labels)
    batch_edge=batch_get_edge(batch_ann_img,edge_width)
    return tf.convert_to_tensor(value=batch_edge,dtype=tf_labels.dtype,name=name)

def get_backbone_index(backbone_name,upsample_ratio):
    """
    backboen_name: resnet50
    """
    backbone=globals()[backbone_name.upper()](
            include_top=False, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
    if backbone_name.lower() in ['vgg16','vgg19']:
        block_pool_name='block%d_pool'%(upsample_ratio+1)
        for idx,l in enumerate(backbone.layers):
            if l.name.startswith(block_pool_name):
                return idx-1
        
        assert upsample_ratio==5,'exception for upsample_ratio=%d'%upsample_ratio 
        return len(backbone.layers)-1
    elif backbone_name.lower() == 'resnet50':
        # if input size is 224
        # layer_output_size=[224,112,55,28,14,7]
        layer_names=['input','conv1','res2c_branch2c','res3d_branch2c','res4f_branch2c','res5c_branch2c']
        layer_offset=[0,2,3,3,3,3]
        
        for idx,l in enumerate(backbone.layers):
            if l.name.startswith(layer_names[upsample_ratio]):
                return idx+layer_offset[upsample_ratio]
            
    elif backbone_name.lower()=='xception':
        # if input size is 224
        # layer_output_size=[224,111,55,28,14,7]
        layer_names=['input','block1_conv1','block3_sepconv2','block4_sepconv2','block13_sepconv2','block14_sepconv2']
        layer_offset=[0,2,1,1,1,2]
        
        for idx,l in enumerate(backbone.layers):
            if l.name.startswith(layer_names[upsample_ratio]):
                return idx+layer_offset[upsample_ratio]
    else:
        assert False,'unknown backone name %s'%backbone_name

    assert False,'exception for backbone name %s and upsample ratio %d'%(backbone_name,upsample_ratio)        
    